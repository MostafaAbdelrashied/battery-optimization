from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from loguru import logger

from battery_management.assets.battery import Battery
from battery_management.assets.charging_point import ChargingPoint
from battery_management.results_handler.site_result import SiteResult
from battery_management.visualization.battery_plots import PlottingTool

np.random.seed(42)


class FleetOptimizationBaseclass(ABC):
    """
    Base Fleet Optimizer.
    """

    def __init__(
        self,
        id: int = None,
        calculate_savings: bool = False,
        fully_charged_as_penalty: bool = False,
        dt: float = 1,
        single_continuous_session_allowed: bool = False,
        penalize_spiky_behaviour: bool = False,
        default_strategy: str = "inactive",
    ):
        """
         The baseclass comes without functionalities,
         it merely provides a skeleton to include new optimizers efficiently

         Parameters
         ----------
         id: str
             identifier for book-keeping

         calculate_savings: bool
             Flag for a lot of post-processing that can be disabled to improve performance e.g. in steering scenarios

         fully_charged_as_penalty: bool
             If true the condition to reach the charging target is imposed as a penalty in the target function, if
             false it is added as a condition in the optimizer. For details see the different implementations

         dt: float
             delta time, the time step for a single data point. Used to convert power into energy

        penalize_spiky_behaviour: bool
             Under some conditions (mostly flat prices), the optimization might not have any incentive to follow a
             "human-sensible" way to optimize. For example, provided that the system is not under tight constraint and
             has to charge 10 kWh in a battery in the next 6 hours, then the optimization can do whatever it likes (
             2.5 in first hour, then 0 then 4, etc.) as it will not change the overall cost. This was criticized in
             the OMNe project as it is then difficult for a person lambda to explain why the optimizer is behaving so.
             As a result, we use this flag to penalize slightly the creation of peaks ( by adding a cost proportional
             to the square of the size of the peak times a factor), so that the algorithm is incentivize to keep the
             power as low as possible over time.

        """
        self.id = id
        self.solve_time = None
        self.objective_value = None
        self.status = None
        self.type = None
        self.n_t = None
        self.id_list = set()
        self.batteries = []

        # ------------------------------------------------
        #  Flags
        # ------------------------------------------------
        self.calculate_savings = calculate_savings
        self.late_charging = None
        self.fully_charged_as_penalty = fully_charged_as_penalty
        self.single_continuous_session_allowed = single_continuous_session_allowed
        self.penalize_spiky_behaviour = penalize_spiky_behaviour
        self.allow_curtailment = False
        self.default_strategy = default_strategy  # Charging Strategy in case of failure ("continuous", "active",
        # "early", "late")

        # ------------------------------------------------
        #  Prices, Marketed Volumes, Flexibility
        # ------------------------------------------------

        # We might use this to assert that Pandas Time Series can be used and are being checked for equal indices
        self.date_range = None
        self.tariffs_import = None
        self.tariffs_export = None
        self.capacity_tariffs_import = None
        self.capacity_tariffs_export = None
        self.triad_tariffs_import = None
        self.triad_tariffs_export = None

        self.include_battery_costs = False

        # We initialize the Mask Array for the marketed volumes. This is used to determine which time steps use
        # prices and which do not due to marketed volumes
        # The total cost is [1-mask_marketed(t)] * cost_from_price(t) + mask_marketed(t) * penalty_marketed(t)
        # Upon adding marketed volumes this will be updated
        self.marketed_volumes = None
        self.mask_marketed = None

        # Flexibility products come with a set of parameters:
        # - Prices for pos/neg availability and a buffer factor (defined by the customer)
        self.prices_flex_neg = None
        self.prices_flex_pos = None
        self.flex_buffer = 1
        self.symmetrical_flex = False

        # Marketed Flexibility
        self.marketed_flex_neg = None
        self.marketed_flex_pos = None
        self.mask_flex_neg = None
        self.mask_flex_pos = None
        self.marketed_flex_coeff = 1

        # Site Load
        self.site_load = None

        # Grid Connection
        self.site_load_restriction_charge = None
        self.site_load_restriction_discharge = None
        self.purchase_efficiency = 1
        self.feed_efficiency = 1
        self.limit_as_penalty = False

        # Do we plot the checkup?
        self.save_file = None

        # Time Discretization
        self.dt = dt

        # Charging Points [OPTIONAL]
        self.charging_points = None

    # ------------------------------------------------------------------
    #  Main Interface
    # ------------------------------------------------------------------
    def add_date_range(self, date_range: pd.date_range):
        if self.n_t is None:
            self.n_t = len(date_range)
        else:
            assert self.n_t == len(date_range)
            self.date_range = date_range

    def add_battery(self, battery: Battery):
        if self.n_t is None:
            self.n_t = len(battery.connected)
            self.mask_marketed = np.array([False] * self.n_t)
        else:
            assert self.n_t == len(battery.connected)

        self.batteries.append(battery)

    # --------------------- Charging Points -----------------------------

    def add_charging_point(self, charging_point: ChargingPoint):
        """
            Like a site can have site limits with regard to its grid connection capacity, so can sites have a limited
            number of charging points which limits the number of batteries which can be charged at the same time.
            Here we save the charging points configuration so that we can later assign batteries to them and so ensure
            that batteries are charged according to the charging points availability.
        Parameters
        ----------
        charging_point

        Returns
        -------

        """
        if not self.charging_points:
            self.charging_points = []
        self.charging_points.append(charging_point)

    def assign_batteries_to_charging_point(self):
        """
            Charging Points have been provided along with batteries, we need to allocate batteries to charging points
            and disconnect batteries which cannot be allocated because of lacking available charging points.

            Note that there will be in the future some intelligence here as charging points might have different
            characteristics (e.g. Fast charging) so that the batteries with biggest energy needs or shortest
            connected time can have access to this chargers. There can also be a policy (flag) that gives priority to
            certain batteries to Fast Charging points (use case: commercial vehicles vs. employee lambda)

            For now find the first available CPs using ChargingPoint.is_available(start, end).
        Returns
        -------

        """
        # Loop over the batteries
        #   find available charging point for battery connected times
        #   if there is one: assign charging point by setting the battery parameters (power_charge_max, etc.) to the
        #                    ones corresponding to the charge point. Also tag the battery and the charge point for
        #                    later retrieval.
        #   otherwise: Battery will not be connected, set battery.connected to False (TODO make sure it does not
        #    break the algorithm by removing adding a check that battery is connected not False before adding to
        #    optimizer constraints)

        if self.charging_points is None:
            message = "Please first add charging points using add_charging_point()."
            logger.error(message)
            return {"message": message}

        # NOTE: Charging Points are time based so we need a date range... What's not elegant is that it's an extra
        # step on the user for now
        if self.date_range is None:
            message = (
                "Cannot assign batteries to charging points without a date range. Please provide a date range "
                "using add_date_range().."
            )
            logger.error(message)
            return {"message": message}

        to_delete = []
        for i, bat in enumerate(self.batteries):
            # If the battery is never connected skip
            # If the battery is stationary skip
            if not bat.is_connected() or bat.stationary:
                continue
            # First make sure that there only one charging session
            assert bat.has_single_charging_session()

            # Maybe we are just re-running the same optimization and the battery was already assigned a CP
            if bat.affected_charging_point_id is not None:
                logger.debug(
                    f"Battery {bat.id} already assigned to charging point {bat.affected_charging_point_id}."
                )
                continue

            _df = pd.DataFrame(
                {"connected": np.int64(bat.connected)}, index=self.date_range
            )

            # Find start and end from date range and connected
            start = _df[_df["connected"] > 0].index[0]
            end = _df[_df["connected"] > 0].index[-1]

            assigned_cp = self.find_first_available_charging_point(
                self.charging_points, start, end
            )

            if assigned_cp is None:
                # We disconnect the battery
                logger.warning(
                    f"Battery {bat.id} cannot be assigned to any charging point and will not be charged."
                )
                bat.connected = [False] * len(bat.connected)

                # or we delete it
                to_delete.append(i)

            else:
                logger.debug(
                    f"Battery {bat.id} assigned to charging point {assigned_cp.asset_id}."
                )
                assigned_cp.book(start, end)
                bat.affected_charging_point_id = assigned_cp.asset_id
                # TODO or is it the min of the one of the battery? Originally took the min but then it stays the
                #  value which was set in the battery and does not allow fast charging for example, unless it is
                #  clearly set to the max power of the battery and not of the expected charging point
                bat.power_charge_max = assigned_cp.charging_power_kw
                bat.power_discharge_max = assigned_cp.discharging_power_kw

        for i in sorted(to_delete, reverse=True):
            del self.batteries[i]

        return {"message": "Success."}

    @staticmethod
    def find_first_available_charging_point(
        charging_points: list, start: pd.Timestamp, end: pd.Timestamp
    ):
        """
            NOTE: This can be removed from this class when a better place is found.
        Parameters
        ----------
        charging_points: list
        start: pd.Timestamp
        end: pd.Timestamp

        Returns
        -------

        """
        available_cps = [cp for cp in charging_points if cp.is_available(start, end)]
        # Here we have room for improvement in the future by selecting most appropriate choice
        if len(available_cps) == 0:
            return None
        return available_cps[0]

    # --------------------- Site -----------------------------
    # TODO: Suggest to replace add_site_limits with add_grid which contains limits and efficiencies and uses the same
    #  vocabulary as class Grid
    # TODO: CHECK/REFACTOR V2G and OMNe
    def add_site_limits(
        self,
        site_load_restriction_charge: float = None,
        site_load_restriction_discharge: float = None,
        limit_as_penalty: bool = None,
    ):
        """

        Parameters
        ----------
        site_load_restriction_charge: float
            positive number that gives a limit on the total charge a site can add per time step
        site_load_restriction_discharge: float
            positive number that gives a limit on the total discharge a site can feed into the grid per time stamp
        limit_as_penalty: bool
            the optimizer can use this as a constraint (flag false) or penalty (flag true)

        Returns
        -------

        """
        if site_load_restriction_charge is not None:
            self.site_load_restriction_charge = site_load_restriction_charge
        if site_load_restriction_discharge is not None:
            self.site_load_restriction_discharge = site_load_restriction_discharge
        if limit_as_penalty is not None:
            self.limit_as_penalty = limit_as_penalty

        return 0

    # TODO: Suggest to move flag limit_as_penalty to normal flag to be activated from the class instance (as other
    #  flags)
    # TODO: Argument should be Grid
    def add_grid(
        self,
        feed_power_limit: float = None,
        purchase_power_limit: float = None,
        purchase_efficiency: float = 1,
        feed_efficiency: float = 1,
        limit_as_penalty: bool = None,
    ):
        """

        Parameters
        ----------
        feed_efficiency
        purchase_efficiency
        purchase_power_limit: float
            positive number that gives a limit on the total charge a site can add per time step
        feed_power_limit: float
            positive number that gives a limit on the total discharge a site can feed into the grid per time stamp

        Returns
        -------

        """
        # TODO: Suggest to also rename the parameters (I would then take the responsibility to update in V2G+OMNe)
        if purchase_power_limit is not None:
            self.site_load_restriction_charge = purchase_power_limit
        if feed_power_limit is not None:
            self.site_load_restriction_discharge = feed_power_limit
        if limit_as_penalty is not None:
            self.limit_as_penalty = limit_as_penalty
        self.purchase_efficiency = purchase_efficiency
        self.feed_efficiency = feed_efficiency
        logger.debug(
            f"Adding Grid limits {self.site_load_restriction_charge}/{self.site_load_restriction_discharge} "
            f"and Efficiencies {self.purchase_efficiency}/{self.feed_efficiency}"
        )
        return 0

    def add_site_load(self, site_load: np.array):
        """
        The POWER consumption of the site where the battery is located. This is crucial for two points:
        1) Prices: do we lower the import on a site or do we actually feed into the grid?
          different prices apply
        2) There may be limits on site level, most prominently a ban to feed into the grid

        Parameters
        ----------
        site_load: np.array of power values
            one value for each time step

        Returns
        -------

        """
        if self.n_t is None:
            self.n_t = len(site_load)
            self.site_load = site_load
        else:
            assert len(site_load) == self.n_t
            self.site_load = site_load

        return 0

    # --------------------- Prices, Volumes, Flex -----------------------------

    def add_prices(
        self,
        tariffs_import: np.array = None,
        tariffs_export: np.array = None,
        capacity_tariffs_import: float = None,
        capacity_tariffs_export: float = None,
        triad_tariffs_import: np.array = None,
        triad_tariffs_export: np.array = None,
    ):
        """
        Add new prices and update the Cost Function
        # ToDo: can we bring the site concept into this? This is not straight-forward! We have to

        Parameters
        ----------
        tariffs_import : np.array
            time series of prices for purchasing energy from the grid
        tariffs_export : np.array
            time series of prices for feeding energy into the grid
        capacity_tariffs_import: float
            fixed price for peak import in time range
        capacity_tariffs_export: float
            fixed price for peak export in time range
        triad_tariffs_import: np.array
            prices for uk-specific triad product
        triad_tariffs_export: np.array
            prices for uk-specific triad product

        Returns
        -------

        Raises
        ------
        """
        if tariffs_import is not None:
            assert (
                len(tariffs_import) == self.n_t
            ), f"{len(tariffs_import)} != {self.n_t}"
            self.tariffs_import = tariffs_import

        if tariffs_export is not None:
            assert len(tariffs_export) == self.n_t
            self.tariffs_export = tariffs_export

        if capacity_tariffs_import is not None:
            self.capacity_tariffs_import = capacity_tariffs_import

        if capacity_tariffs_export is not None:
            self.capacity_tariffs_export = capacity_tariffs_export

        if triad_tariffs_import is not None:
            assert (
                len(triad_tariffs_import) == self.n_t
            ), f"{len(triad_tariffs_import)} != {self.n_t}"
            self.triad_tariffs_import = triad_tariffs_import

        if triad_tariffs_export is not None:
            assert (
                len(triad_tariffs_export) == self.n_t
            ), f"{len(triad_tariffs_export)} != {self.n_t}"
            self.triad_tariffs_export = triad_tariffs_export

        return 0

    def add_marketed_volumes(self, marketed_volumes: np.array):
        """
        Previously marketed volumes that affect the optimization target function: for time steps where marketed
        volumes exist the cost from energy becomes irrelevant, only the absolute difference between energy and
        marketed energy is used as a penalty

        Parameters
        ----------
        marketed_volumes: np.array
            Time series for marketed volumes. Notice that no marketed volumes need to be added as np.nan. Zeros are not
            possible as this would be a zero that needs to be matched


        Returns
        -------

        """
        assert len(marketed_volumes) == self.n_t, (
            f"len(marketed_volumes): {len(marketed_volumes)}, " f"self.n_t: {self.n_t}"
        )
        self.marketed_volumes = marketed_volumes
        self.mask_marketed = np.isfinite(marketed_volumes)

        return 0

    def add_flex(
        self,
        prices_flex_pos: np.array = None,
        prices_flex_neg: np.array = None,
        flex_buffer: float = 1,
        symmetrical_flex: bool = False,
    ):
        """
        We add flexibility forecast
        This determines how much flexibility (+/-) can be marketed at each time step
        Usually this only applies to a limited part of the whole period.
        This has to be taken into account via the following:
        - Prices where no flex is marketed need to be zero
        - Only prices where flex is marketed are non-vanishing
        This means we do not need to pass a True/False mask or similar in this instance

        Parameters
        ----------
        symmetrical_flex: Whether the flexibility positive and negative should be equal (symmetrical) or not
        prices_flex_pos
        prices_flex_neg
        flex_buffer

        Returns
        -------

        """
        self.flex_buffer = flex_buffer
        self.symmetrical_flex = symmetrical_flex

        if prices_flex_pos is not None:
            self.mask_flex_pos = np.isfinite(prices_flex_pos) * 1
            if self.n_t is None:
                self.prices_flex_pos = prices_flex_pos
            else:
                assert len(prices_flex_pos) == self.n_t
                self.prices_flex_pos = prices_flex_pos

        if prices_flex_neg is not None:
            self.mask_flex_neg = np.isfinite(prices_flex_neg) * 1
            if self.n_t is None:
                self.prices_flex_neg = prices_flex_neg
            else:
                assert len(prices_flex_neg) == self.n_t
                self.prices_flex_neg = prices_flex_neg

        # If only one price info as been given and symmetrical is used, then affect the same price to the missing one
        if symmetrical_flex and (prices_flex_neg is None or prices_flex_pos is None):
            if self.prices_flex_pos is None:
                self.prices_flex_pos = self.prices_flex_neg
            if self.prices_flex_neg is None:
                self.prices_flex_neg = self.prices_flex_pos

    def add_marketed_flex(
        self,
        pos_flex: np.array = None,
        neg_flex: np.array = None,
        coefficient: float = 1,
    ):
        self.marketed_flex_coeff = coefficient
        if pos_flex is not None:
            self.mask_flex_pos = np.isfinite(pos_flex) * 1
            if self.n_t is None:
                self.marketed_flex_pos = pos_flex
            else:
                assert len(pos_flex) == self.n_t
                self.marketed_flex_pos = pos_flex

        if neg_flex is not None:
            self.mask_flex_neg = np.isfinite(neg_flex) * 1
            if self.n_t is None:
                self.marketed_flex_neg = neg_flex
            else:
                assert len(neg_flex) == self.n_t
                self.marketed_flex_neg = neg_flex

    # =========================================================================================
    #  Optimization
    # =========================================================================================
    #
    @abstractmethod
    def optimize(self):
        pass

    # =========================================================================================
    #  Plotting
    # =========================================================================================
    def plot(self, result: SiteResult, filename: str):
        PlottingTool._plot_v2g(result, filename)
