from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from battery_management.assets.battery import Battery
from battery_management.assets.charging_point import ChargingPoint
from battery_management.helper.site_result import SiteResult
from battery_management.helper.visualization import PlottingTool

np.random.seed(42)


class FleetOptimizationBaseclass(ABC):
    """
    Base Fleet Optimizer.

    Parameters
    ----------
    id : int, optional
        Identifier for book-keeping.
    calculate_savings : bool, default False
        Flag for post-processing that can be disabled to improve performance.
    fully_charged_as_penalty : bool, default False
        If True, the condition to reach the charging target is imposed as a penalty in the target function.
    dt : float, default 1
        Delta time, the time step for a single data point. Used to convert power into energy.
    single_continuous_session_allowed : bool, default False
        Whether a single continuous charging session is allowed.
    penalize_spiky_behaviour : bool, default False
        If True, penalizes the creation of power peaks.
    default_strategy : str, default "inactive"
        Charging strategy in case of failure ("continuous", "active", "early", "late").
    """

    def __init__(
        self,
        id: Optional[int] = None,
        calculate_savings: bool = False,
        fully_charged_as_penalty: bool = False,
        dt: float = 1,
        single_continuous_session_allowed: bool = False,
        penalize_spiky_behaviour: bool = False,
        default_strategy: str = "inactive",
    ):
        self.id = id
        self.solve_time: Optional[float] = None
        self.objective_value: Optional[float] = None
        self.status: Optional[str] = None
        self.type: Optional[str] = None
        self.n_t: Optional[int] = None
        self.id_list: set = set()
        self.batteries: List[Battery] = []

        self.calculate_savings = calculate_savings
        self.late_charging: Optional[bool] = None
        self.fully_charged_as_penalty = fully_charged_as_penalty
        self.single_continuous_session_allowed = single_continuous_session_allowed
        self.penalize_spiky_behaviour = penalize_spiky_behaviour
        self.allow_curtailment = False
        self.default_strategy = default_strategy

        self.date_range: Optional[pd.DatetimeIndex] = None
        self.tariffs_import: Optional[np.ndarray] = None
        self.tariffs_export: Optional[np.ndarray] = None
        self.capacity_tariffs_import: Optional[float] = None
        self.capacity_tariffs_export: Optional[float] = None
        self.triad_tariffs_import: Optional[np.ndarray] = None
        self.triad_tariffs_export: Optional[np.ndarray] = None

        self.include_battery_costs = False

        self.marketed_volumes: Optional[np.ndarray] = None
        self.mask_marketed: Optional[np.ndarray] = None

        self.prices_flex_neg: Optional[np.ndarray] = None
        self.prices_flex_pos: Optional[np.ndarray] = None
        self.flex_buffer = 1
        self.symmetrical_flex = False

        self.marketed_flex_neg: Optional[np.ndarray] = None
        self.marketed_flex_pos: Optional[np.ndarray] = None
        self.mask_flex_neg: Optional[np.ndarray] = None
        self.mask_flex_pos: Optional[np.ndarray] = None
        self.marketed_flex_coeff = 1

        self.site_load: Optional[np.ndarray] = None

        self.site_load_restriction_charge: Optional[float] = None
        self.site_load_restriction_discharge: Optional[float] = None
        self.purchase_efficiency = 1
        self.feed_efficiency = 1
        self.limit_as_penalty = False

        self.save_file: Optional[str] = None

        self.dt = dt

        self.charging_points: Optional[List[ChargingPoint]] = None

    def add_date_range(self, date_range: pd.DatetimeIndex) -> None:
        """
        Add date range to the optimizer.

        Parameters
        ----------
        date_range : pd.DatetimeIndex
            Date range for the optimization.
        """
        if self.n_t is None:
            self.n_t = len(date_range)
        else:
            assert self.n_t == len(date_range)
        self.date_range = date_range

    def add_battery(self, battery: Battery) -> None:
        """
        Add a battery to the optimizer.

        Parameters
        ----------
        battery : Battery
            Battery to be added.
        """
        if self.n_t is None:
            self.n_t = len(battery.connected)
            self.mask_marketed = np.array([False] * self.n_t)
        else:
            assert self.n_t == len(battery.connected)

        self.batteries.append(battery)

    def add_charging_point(self, charging_point: ChargingPoint) -> None:
        """
        Add a charging point to the optimizer.

        Parameters
        ----------
        charging_point : ChargingPoint
            Charging point to be added.
        """
        if not self.charging_points:
            self.charging_points = []
        self.charging_points.append(charging_point)

    def assign_batteries_to_charging_point(self) -> dict:
        """
        Assign batteries to available charging points.

        Returns
        -------
        dict
            A dictionary containing a message about the assignment process.
        """
        if self.charging_points is None:
            message = "Please first add charging points using add_charging_point()."
            logger.error(message)
            return {"message": message}

        if self.date_range is None:
            message = "Cannot assign batteries to charging points without a date range. Please provide a date range using add_date_range()."
            logger.error(message)
            return {"message": message}

        to_delete = []
        for i, bat in enumerate(self.batteries):
            if not bat.is_connected() or bat.stationary:
                continue

            assert bat.has_single_charging_session()

            if bat.affected_charging_point_id is not None:
                logger.debug(
                    f"Battery {bat.id} already assigned to charging point {bat.affected_charging_point_id}."
                )
                continue

            _df = pd.DataFrame(
                {"connected": np.int64(bat.connected)}, index=self.date_range
            )
            start = _df[_df["connected"] > 0].index[0]
            end = _df[_df["connected"] > 0].index[-1]

            assigned_cp = self.find_first_available_charging_point(
                self.charging_points, start, end
            )

            if assigned_cp is None:
                logger.warning(
                    f"Battery {bat.id} cannot be assigned to any charging point and will not be charged."
                )
                bat.connected = [False] * len(bat.connected)
                to_delete.append(i)
            else:
                logger.debug(
                    f"Battery {bat.id} assigned to charging point {assigned_cp.asset_id}."
                )
                assigned_cp.book(start, end)
                bat.affected_charging_point_id = assigned_cp.asset_id
                bat.power_charge_max = assigned_cp.charging_power_kw
                bat.power_discharge_max = assigned_cp.discharging_power_kw

        for i in sorted(to_delete, reverse=True):
            del self.batteries[i]

        return {"message": "Success."}

    @staticmethod
    def find_first_available_charging_point(
        charging_points: List[ChargingPoint], start: pd.Timestamp, end: pd.Timestamp
    ) -> Optional[ChargingPoint]:
        """
        Find the first available charging point for a given time range.

        Parameters
        ----------
        charging_points : List[ChargingPoint]
            List of charging points to search.
        start : pd.Timestamp
            Start time of the charging session.
        end : pd.Timestamp
            End time of the charging session.

        Returns
        -------
        Optional[ChargingPoint]
            The first available charging point, or None if none are available.
        """
        available_cps = [cp for cp in charging_points if cp.is_available(start, end)]
        return available_cps[0] if available_cps else None

    def add_grid(
        self,
        feed_power_limit: Optional[float] = None,
        purchase_power_limit: Optional[float] = None,
        purchase_efficiency: float = 1,
        feed_efficiency: float = 1,
        limit_as_penalty: Optional[bool] = None,
    ) -> int:
        """
        Add grid parameters to the optimizer.

        Parameters
        ----------
        feed_power_limit : float, optional
            Limit on the total discharge a site can feed into the grid per time step.
        purchase_power_limit : float, optional
            Limit on the total charge a site can add per time step.
        purchase_efficiency : float, default 1
            Efficiency of purchasing power from the grid.
        feed_efficiency : float, default 1
            Efficiency of feeding power into the grid.
        limit_as_penalty : bool, optional
            Whether to use the limit as a penalty or constraint.

        Returns
        -------
        int
            Status code (0 for success).
        """
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

    def add_site_load(self, site_load: np.ndarray) -> int:
        """
        Add site load to the optimizer.

        Parameters
        ----------
        site_load : np.ndarray
            Power consumption of the site for each time step.

        Returns
        -------
        int
            Status code (0 for success).
        """
        if self.n_t is None:
            self.n_t = len(site_load)
            self.site_load = site_load
        else:
            assert len(site_load) == self.n_t
            self.site_load = site_load

        return 0

    def add_prices(
        self,
        tariffs_import: Optional[np.ndarray] = None,
        tariffs_export: Optional[np.ndarray] = None,
        capacity_tariffs_import: Optional[float] = None,
        capacity_tariffs_export: Optional[float] = None,
        triad_tariffs_import: Optional[np.ndarray] = None,
        triad_tariffs_export: Optional[np.ndarray] = None,
    ) -> int:
        """
        Add prices to the optimizer.

        Parameters
        ----------
        tariffs_import : np.ndarray, optional
            Time series of prices for purchasing energy from the grid.
        tariffs_export : np.ndarray, optional
            Time series of prices for feeding energy into the grid.
        capacity_tariffs_import : float, optional
            Fixed price for peak import in time range.
        capacity_tariffs_export : float, optional
            Fixed price for peak export in time range.
        triad_tariffs_import : np.ndarray, optional
            Prices for UK-specific triad product (import).
        triad_tariffs_export : np.ndarray, optional
            Prices for UK-specific triad product (export).

        Returns
        -------
        int
            Status code (0 for success).
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

    def add_marketed_volumes(self, marketed_volumes: np.ndarray) -> int:
        """
        Add marketed volumes to the optimizer.

        Parameters
        ----------
        marketed_volumes : np.ndarray
            Time series for marketed volumes. NaN values indicate no marketed volumes.

        Returns
        -------
        int
            Status code (0 for success).
        """
        assert (
            len(marketed_volumes) == self.n_t
        ), f"len(marketed_volumes): {len(marketed_volumes)}, self.n_t: {self.n_t}"
        self.marketed_volumes = marketed_volumes
        self.mask_marketed = np.isfinite(marketed_volumes)

        return 0

    def add_flex(
        self,
        prices_flex_pos: Optional[np.ndarray] = None,
        prices_flex_neg: Optional[np.ndarray] = None,
        flex_buffer: float = 1,
        symmetrical_flex: bool = False,
    ) -> None:
        """
        Add flexibility forecast to the optimizer.

        Parameters
        ----------
        prices_flex_pos : np.ndarray, optional
            Prices for positive flexibility.
        prices_flex_neg : np.ndarray, optional
            Prices for negative flexibility.
        flex_buffer : float, default 1
            Buffer for flexibility.
        symmetrical_flex : bool, default False
            Whether the flexibility positive and negative should be equal.
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

        if symmetrical_flex and (prices_flex_neg is None or prices_flex_pos is None):
            if self.prices_flex_pos is None:
                self.prices_flex_pos = self.prices_flex_neg
            if self.prices_flex_neg is None:
                self.prices_flex_neg = self.prices_flex_pos

    def add_marketed_flex(
        self,
        pos_flex: Optional[np.ndarray] = None,
        neg_flex: Optional[np.ndarray] = None,
        coefficient: float = 1,
    ) -> None:
        """
        Add marketed flexibility to the optimizer.

        Parameters
        ----------
        pos_flex : np.ndarray, optional
            Positive flexibility values.
        neg_flex : np.ndarray, optional
            Negative flexibility values.
        coefficient : float, default 1
            Coefficient for marketed flexibility.
        """
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

    @abstractmethod
    def optimize(self) -> None:
        """
        Perform the optimization.
        This method should be implemented by subclasses.
        """
        pass

    def plot(self, result: SiteResult, filename: str) -> None:
        """
        Plot the optimization results.

        Parameters
        ----------
        result : SiteResult
            The result of the optimization to be plotted.
        filename : str
            The filename to save the plot.
        """
        PlottingTool._plot_v2g(result, filename)
