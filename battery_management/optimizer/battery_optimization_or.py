import os

import numpy as np
import pandas as pd
from loguru import logger
from ortools.linear_solver import pywraplp

from battery_management.assets.battery import Battery
from battery_management.optimizer.battery_optimization_baseclass import (
    FleetOptimizationBaseclass,
)
from battery_management.results_handler.site_result import SiteResult

np.random.seed(42)


class FleetOptimizationOR(FleetOptimizationBaseclass):
    """
    ORTools implementation of the FleetOptimization. This is curently the only implementation fully up-to-date.
    """

    def __init__(self, *args, **kwargs):
        """

        Parameters
        ----------
        id: int
            The ID can be used when handling more than one optimization at the time
        dt: int
            Time discretization to convert power into energy.
            If dt=1, then it can be seen as using energy kwh (v2g case).
            If using power and time interval is 15min (eva case) then dt should be 0.25.

        args
        kwargs
        """
        super().__init__(**kwargs)

        self.type = "OR"

        self.id_list = set()
        self.batteries = []

        # Create Parameter Shells

        self.grid_power = {"feed": {}, "purchase": {}, "curtail": {}}
        self.grid_peak = {"feed": None, "purchase": None}
        self.fleet_power = {}  # first key will be vehicle_id, then "charge" and "discharge"
        self.fleet_energy = {}  # first key will be vehicle_id, then soc_kwh

        self.grid_bool = {"feed": {}, "purchase": {}, "curtail": {}}
        self.fleet_bool = {}  # first key will be vehicle_id, then "charge" and "discharge"
        self.sum_power_fleet = {"charge": {}, "discharge": {}}

        self.site_constraint = {"feed": None, "purchase": None}

        # Flexibility Variables
        self.flex_pos = {}
        self.flex_neg = {}

        # Collect Cost components in dictionary
        self.cost = {}

        self.limit_purchase_cost = None
        self.limit_purchase_penalty = 5000
        self.fully_charged_penalty = 1000
        self.spiky_behaviour_penalty = (
            0.01  # cf. description of flag penalize_spiky_behaviour in baseclass. this
        )
        # penalty would be maybe better under the base class as well, but it is useful to have it here next to the
        # other penalties for comparison
        # NOTE: spiky_behaviour_penalty could be interacting with an optimization over low prices and so should stay low
        # ideally lower than the prices

        self.model = None
        self.results = {}

    # ---------------------------------------------------
    #  Initialize Variables
    # ---------------------------------------------------

    def _initialize_grid_power(self):
        """
        Create the grid power variables for each time step:
        - Grid Power [kW]
        - Grid Boolean Do we charge from the grid or not?
        """

        for t in range(self.n_t):
            for key in self.grid_power:
                label = f"grid_power_{key}_{t}"
                if not self.allow_curtailment and key == "curtail":
                    upper_limit = 0
                else:
                    upper_limit = self.model.infinity()
                self.grid_power[key][t] = self.model.NumVar(0, upper_limit, label)

            for key in self.grid_bool:
                label = f"grid_bool_{key}_{t}"
                self.grid_bool[key][t] = self.model.BoolVar(label)

            # The cleanest way to disable curtailment is to force the boolean to be zero
            if not self.allow_curtailment:
                self.model.Add(
                    self.grid_bool["curtail"][t] == 0, f"disable_curtailment_{t}"
                )

        return 0

    def _initialize_battery(self, battery: Battery):
        """
        Initialize the power and energy variables for the individual batteries and charger
        Conventions are:
        - Efficiency: Efficiency is applied when calculating the grid power, i.e. for the charger to provide
            the maximum of e.g. 10kW it needs to take 10 kW / 0.95 for an efficiency of 95%
        - Power:

        Parameters
        ----------
        battery

        Returns
        -------

        """
        # Maybe the battery could not be allocated to a charge point so we should not add it to the optimizer
        if not battery.is_connected():
            return {"message": f"Battery {battery.id} is not connected."}

        i = battery.id

        # ----------------------------------------------------------------------
        #  Battery Variables for Charging/Discharging
        # ----------------------------------------------------------------------

        self.fleet_power[i] = {"charge": {}, "discharge": {}}
        self.fleet_energy[i] = {}
        self.fleet_bool[i] = {"charge": {}, "discharge": {}, "disconnected": {}}
        for t in range(self.n_t):
            # (Dis-)charge Variables
            self.fleet_power[i]["charge"][t] = self.model.NumVar(
                0,
                battery.power_charge_max if battery.connected[t] else 0,
                f"power_{i}_charge_{t}",
            )

            self.fleet_power[i]["discharge"][t] = self.model.NumVar(
                0,
                battery.power_discharge_max if battery.connected[t] else 0,
                f"power_{i}_discharge_{t}",
            )

            self.fleet_energy[i][t] = self.model.NumVar(
                battery.energy_min, battery.energy_max, f"energy_{i}_{t}"
            )

            # (Dis-)charge Booleans (ensure only one is ever used)
            # Disconnect Flag allows to stop the charging session (and so setting power to 0 when charging_power_min>0)
            self.fleet_bool[i]["charge"][t] = self.model.BoolVar(f"bool_{i}_charge_{t}")
            self.fleet_bool[i]["discharge"][t] = self.model.BoolVar(
                f"bool_{i}_discharge_{t}"
            )
            self.fleet_bool[i]["disconnected"][t] = self.model.BoolVar(
                f"bool_{i}_disconnected_{t}"
            )

        # ----------------------------------------------------------------------
        #  Current energy per Vehicle (so far we have only considered deltas)
        #  and Charging Targets
        # ----------------------------------------------------------------------

        # The change in battery status from power depends on the efficiency. Values around 95% are typical
        self.fleet_energy[i][0] = (
            battery.energy_start
            + self.fleet_power[i]["charge"][0] * self.dt
            - self.fleet_power[i]["discharge"][0] * self.dt
        )
        self.model.Add(
            self.fleet_energy[i][0] >= battery.energy_min,
            f"energy_min_constraint_{i}_{0}",
        )
        self.model.Add(
            self.fleet_energy[i][0] <= battery.energy_max,
            f"energy_max_constraint_{i}_{0}",
        )

        for t in range(1, self.n_t):
            self.fleet_energy[i][t] = self.fleet_energy[i][t - 1] + (
                self.fleet_power[i]["charge"][t] * self.dt
                - self.fleet_power[i]["discharge"][t] * self.dt
            )

            self.model.Add(
                self.fleet_energy[i][t] >= battery.energy_min,
                f"energy_min_constraint_{i}_{t}",
            )
            self.model.Add(
                self.fleet_energy[i][t] <= battery.energy_max,
                f"energy_max_constraint_{i}_{t}",
            )

        # Stationary batteries do not require to be charged to a certain amount at the end of the period
        # In contrario, non-stationary batteries (EVs) have to be charged at the end of the period to a certain level
        # Note that even using "fully_charged_as_penalty" and energy_end=energy_min will constrain the optimization
        # to charge the battery because it is rewarding to do so (as per definition).
        if not battery.stationary:
            if not self.fully_charged_as_penalty:
                # Fully charged is a constraint
                self.model.Add(
                    self.fleet_energy[i][self.n_t - 1] >= battery.energy_end,
                    f"fully_charged_constraint_{i}",
                )
            else:
                # Fully charged is taken as a penalty
                # IMPORTANT NOTE: the current implementation will always try to reach battery energy_max and not battery
                # energy_end because it gets rewarded for charging more than battery energy_end
                remaining_energy_not_charged = (
                    battery.energy_end - self.fleet_energy[i][self.n_t - 1]
                )

                if "fully_charged_penalty" not in self.cost.keys():
                    # This needs to be initialized because it will be iteratively append values for each battery
                    self.cost["fully_charged_penalty"] = []

                self.cost["fully_charged_penalty"].append(
                    remaining_energy_not_charged * self.fully_charged_penalty
                )

        # ----------------------------------------------------------------------
        # Prevent simultaneous charging and discharging of the battery
        # ----------------------------------------------------------------------
        first_time_connected = True
        for t in range(self.n_t):
            self.model.Add(
                self.fleet_power[i]["charge"][t]
                <= self.fleet_bool[i]["charge"][t] * battery.power_charge_max,
                f"maximum_limit_battery_charging_{i}_{t}",
            )
            self.model.Add(
                self.fleet_power[i]["discharge"][t]
                <= self.fleet_bool[i]["discharge"][t] * battery.power_discharge_max,
                f"maximum_limit_battery_discharging_{i}_{t}",
            )
            self.model.Add(
                self.fleet_power[i]["charge"][t]
                >= (1 - self.fleet_bool[i]["disconnected"][t])
                * battery.power_charge_min,
                f"minimum_limit_battery_charging_{i}_{t}",
            )
            self.model.Add(
                self.fleet_bool[i]["charge"][t]
                + self.fleet_bool[i]["discharge"][t]
                + self.fleet_bool[i]["disconnected"][t]
                <= 1,
                f"prevent_simultaneous_charging_and_discharging_{i}_{t}",
            )

            # To prevent the optimizer from using the possibility to be in "discharge" mode to set the power to 0
            # when a power_charge_min is provided, we constraint the boolean allowing the discharging to be 0.
            if battery.power_discharge_max == 0:
                self.model.Add(
                    self.fleet_bool[i]["discharge"][t] == 0,
                    f"cannot_discharge_constraint_{i}_{t}",
                )

            # Setup to allow only one continuous charging session
            # Background: Most EVs have to be charged with a minimum current of 6 Amps (around 1.3kW) and if the power
            # is set to 0kW then the vehicle is considered disconnected and the charging session cannot continue without
            # manually plugging out and in the vehicle to the charging point
            # Implementation: A "disconnect" flag is used which can only be set once and for all to True (=1)

            # First, security for every case, if the battery is not connected then the flag disconnected can be set
            # to true (note that there is a second mecanism which already prevent the power from being different to 0
            # if the battery is not connected
            if not battery.connected[t]:
                self.model.Add(
                    self.fleet_bool[i]["disconnected"][t] == 1,
                    f"disconnected_constraint_{i}_{t}",
                )

            # Only if single_continuous_session_allowed is activated, then we add additional constraints to the
            # "disconnected" flag. These constraints set the first timestep to "connected"=0 and then only allow the
            # following variables to be greater or equal so that:
            # - if 0 at t-1 then [1,0] are possible
            # - if 1 at t-1 then only 1 is possible, so the vehicle is permanently disconnected
            elif self.single_continuous_session_allowed and first_time_connected:
                self.model.Add(
                    self.fleet_bool[i]["disconnected"][t] == 0,
                    f"disconnected_constraint_{i}_{t}",
                )
                first_time_connected = False
            elif self.single_continuous_session_allowed:
                self.model.Add(
                    self.fleet_bool[i]["disconnected"][t]
                    >= self.fleet_bool[i]["disconnected"][t - 1],
                    f"disconnected_constraint_{i}_{t}",
                )

        return 0

    def _initialize_fleet_power(self):
        # ----------------------------------------------------------------------
        # Overall electricity balance
        # ----------------------------------------------------------------------
        # Define abbreviations for electricity sources and sinks
        # NOTE: All of these should be in kW if self.dt!=1 (INCLUDING site_load)
        for t in range(self.n_t):
            # Here is where we need to adjust for the efficiency:
            source = self.grid_power["purchase"][
                t
            ] * self.purchase_efficiency + self.model.Sum(
                [
                    self.fleet_power[battery.id]["discharge"][t]
                    * battery.efficiency_discharge
                    for battery in self.batteries
                ]
            )

            batteries_charge = [
                self.fleet_power[battery.id]["charge"][t] / battery.efficiency_charge
                for battery in self.batteries
            ]
            sink = (
                self.grid_power["feed"][t] / self.feed_efficiency
                + self.grid_power["curtail"][t]
                + self.model.Sum(batteries_charge)
            )

            if self.site_load is not None:
                sl = self.site_load[t]
                if sl > 0:
                    sink += sl
                else:
                    source -= sl

            self.model.Add(source == sink, f"overall_electricity_balance_{t}")

        # ----------------------------------------------------------------------
        # Prevent simultaneous feed-in and purchase from the grid
        # ----------------------------------------------------------------------

        for t in range(self.n_t):
            self.model.Add(
                self.grid_bool["feed"][t]
                + self.grid_bool["curtail"][t]
                + self.grid_bool["purchase"][t]
                == 1,
                f"prevent_simultaneous_feed_in_and_purchase_{t}",
            )

        # ----------------------------------------------------------------------
        #  Totals for Charge/Discharge <=> sum over all assets
        # ----------------------------------------------------------------------
        # This needs to be updated after every battery
        # Since no variables are affected this can be done

        for t in range(self.n_t):
            self.sum_power_fleet["charge"][t] = self.model.Sum(
                [
                    self.fleet_power[battery.id]["charge"][t]
                    for battery in self.batteries
                ]
            )
            self.sum_power_fleet["discharge"][t] = self.model.Sum(
                [
                    self.fleet_power[battery.id]["discharge"][t]
                    for battery in self.batteries
                ]
            )

        return 0

    def _calc_site_limits(self):
        """
        In case of site limits for charging or dis-charging we can add these to the optimizer as constraints in two ways
        a) As a constraint where anything that exceeds the limit is multiplied with a factor and added into the
            target function
        b) As a hard constraint in the optimizer. This might cause some impossible solutions that could be avoided
            in a). If a solution is possible this is the cleaner way

        Parameters
        ----------

        Returns
        -------

        """
        if self.limit_as_penalty:
            # This was written by Quentin for EVA. Is there a reason we only have a restriction for charging?
            # This is the more likely of the two cases, yet not the only option
            if self.site_load_restriction_charge is not None:
                # Create a new variable representing the grid import constraint
                self.site_constraint["purchase"] = self.model.NumVar(
                    self.site_load_restriction_charge,
                    self.model.infinity(),
                    "site_constraint_purchase",
                )
                # Add constraint that the grid imports should always be below the constraint
                for t in range(self.n_t):
                    self.model.Add(
                        self.grid_power["purchase"][t]
                        <= self.site_constraint["purchase"],
                        f"site_import_constraint_{t}",
                    )

                # Heavily penalize increase in the grid import constraint variable
                self.cost["site_limit_purchase"] = [
                    (
                        self.site_constraint["purchase"]
                        - self.site_load_restriction_charge
                    )
                    * self.limit_purchase_penalty,
                ]
        else:
            for t in range(self.n_t):
                if self.site_load_restriction_discharge is not None:
                    self.model.Add(
                        self.grid_power["feed"][t]
                        <= self.grid_bool["feed"][t]
                        * self.site_load_restriction_discharge,
                        f"maximum_limit_feed_in_{t}",
                    )
                if self.site_load_restriction_charge is not None:
                    self.model.Add(
                        self.grid_power["purchase"][t]
                        <= self.grid_bool["purchase"][t]
                        * self.site_load_restriction_charge,
                        f"maximum_limit_purchase_{t}",
                    )

        return 0

    # ---------------------------------------------------
    #  Cost Components Price, marketed volumes (General)
    # ---------------------------------------------------

    def _calc_cost_prices(self):
        """
        Calculate the penalty from energy prices. This is relevant only if prices are given and at the same time
        no marketed volumes that need to be matched. The result is stored in the cost dictionary

        Returns
        -------

        """
        # --------------------------------------------------------------------------------------------
        #  Formulation without cost variables
        # --------------------------------------------------------------------------------------------
        # If there are marketed volumes the cost function in those time steps apply, cost function from energy
        # prices is zero
        if self.mask_marketed is None:
            mm = np.array([0] * self.n_t)
        else:
            mm = self.mask_marketed

        # At time steps where the energy prices apply we need to make sure we consider the site load correctly.
        # The grid power variable is calculated including site load. Hence we need to subtract the site load base line
        # cost here. These extra steps are necessary to make sure we use the correct price if the discharging battery
        # reduces the site load or is fed into the grid
        if self.site_load is not None:
            site_load_pos = np.where(self.site_load > 0, self.site_load, 0)
            site_load_neg = np.where(self.site_load < 0, -self.site_load, 0)
        else:
            site_load_pos = [0] * self.n_t
            site_load_neg = [0] * self.n_t

        if self.tariffs_import is not None:
            cost_import = [
                (1 - mm[t])
                * (
                    self.tariffs_import[t]
                    * (self.grid_power["purchase"][t] - site_load_pos[t])
                    * self.dt
                )
                for t in range(self.n_t)
            ]
        else:
            cost_import = [0 for _ in range(self.n_t)]

        if self.tariffs_export is not None:
            cost_export = [
                -(1 - mm[t])
                * (
                    self.tariffs_export[t]
                    * (self.grid_power["feed"][t] - site_load_neg[t])
                    * self.dt
                )
                for t in range(self.n_t)
            ]
        else:
            cost_export = [0 for _ in range(self.n_t)]

        self.cost["Spot"] = [
            self.model.NumVar(
                -self.model.infinity(), self.model.infinity(), name=f"cost_tariffs_{t}"
            )
            for t in range(self.n_t)
        ]

        if (self.tariffs_export is not None) and (self.tariffs_import is not None):
            _ = [
                self.model.Add(
                    self.cost["Spot"][t] == cost_import[t] + cost_export[t],
                    f"const_tariff_var_{t}",
                )
                for t in range(self.n_t)
            ]
        elif self.tariffs_export is not None:
            _ = [
                self.model.Add(
                    self.cost["Spot"][t] == cost_export[t], f"const_tariff_var_{t}"
                )
                for t in range(self.n_t)
            ]
        elif self.tariffs_import is not None:
            _ = [
                self.model.Add(
                    self.cost["Spot"][t] == cost_import[t], f"const_tariff_var_{t}"
                )
                for t in range(self.n_t)
            ]

        return 0

    def _calc_capacity_prices(self):
        """
        Price Component originally introduced for EVA

        Returns
        -------

        """
        # cost_peaks = costs["peak_fee"] * peak["delta_peak_purchase_positive"]
        # --------------------------------------------------------------------------------------------
        #  Add Variable and Constraint for the peak load
        # --------------------------------------------------------------------------------------------
        if self.capacity_tariffs_import is not None:
            self.grid_peak["purchase"] = self.model.NumVar(
                0, self.model.infinity(), "peak_purchase"
            )
            for t in range(self.n_t):
                self.model.Add(
                    self.grid_power["purchase"][t] <= self.grid_peak["purchase"],
                    f"constraint_peak_purchase_{t}",
                )
            self.cost["Capacity_Purchase"] = [
                self.capacity_tariffs_import * self.grid_peak["purchase"],
            ]

        if self.capacity_tariffs_export is not None:
            self.grid_peak["feed"] = self.model.NumVar(
                0, self.model.infinity(), "peak_feed"
            )
            for t in range(self.n_t):
                self.model.Add(
                    self.grid_power["feed"][t] <= self.grid_peak["feed"],
                    f"constraint_peak_feed_{t}",
                )

            self.cost["Capacity_Feed"] = [
                self.capacity_tariffs_export * self.grid_peak["feed"],
            ]

        return 0

    def _calc_cost_marketed_volumes(self):
        """
        Calculate the penalty for matching marketed volumes (we calculate the absolute, so a perfect match would
        correspond to a zero.
        The result is stored in the cost dictionary

        Returns
        -------

        """

        # --------------------------------------------
        #  Create new constraints
        # --------------------------------------------

        # We add Sum_t |marketed(t) - DeltaE(t)| to the cost function
        # This is not straight-forward in linear programming but can be done:
        # - split into positive and negative part, add the sum of the two to the cost

        diff_pos = [
            self.model.NumVar(
                0, self.model.infinity(), name=f"marketed_volumes_Diff_Pos_{t}"
            )
            for t in range(self.n_t)
        ]
        diff_neg = [
            self.model.NumVar(
                0, self.model.infinity(), name=f"marketed_volumes_Diff_Neg_{t}"
            )
            for t in range(self.n_t)
        ]

        # Notice that diff was only added for readability.
        diff = [0] * self.n_t
        for t in range(self.n_t):
            if self.mask_marketed[t]:
                # Marketed Volumes are in kWh, sum_power_fleet in kW
                diff[t] = self.marketed_volumes[t] - self.dt * (
                    self.sum_power_fleet["charge"][t]
                    - self.sum_power_fleet["discharge"][t]
                )
            else:
                diff[t] = 0
            self.model.Add(
                diff[t] == diff_pos[t] - diff_neg[t], f"marketed_volumes_diff_{t}"
            )

        # ---------------------------------------------------------------------------------------------------------
        #  Formulation with cost variables
        # ---------------------------------------------------------------------------------------------------------
        # # We add this variable for debugging purposes. This way we have direct access to the individual components
        # # of the cost. However, this comes at a cost as we have to add variables and constraints
        # cost_marketed = [self.model.add_var(var_type=CONTINUOUS, lb=0, ub=10000, name=f'cost_marketed_{t}')
        #                  for t in range(self.n_t)]
        #
        # for t in range(self.n_t):
        #     self.model += (cost_marketed[t] == 10 * (diff_pos[t] + diff_neg[t]), f'constr_cost_marketed_{t}')
        #
        # self.cost['MarketedVolumes'] = cost_marketed
        #
        # # Re-run the prices to update the mask for marketed volumes in the cost function
        # self._calc_cost_prices()

        # ---------------------------------------------------------------------------------------------------------
        #  Formulation without cost variables
        # ---------------------------------------------------------------------------------------------------------

        self.cost["MarketedVolumes"] = [
            10 * (diff_pos[t] + diff_neg[t]) for t in range(self.n_t)
        ]

        return 0

    def _calc_cost_spiky_behaviour(self):
        """
            Under some conditions (mostly flat prices), the optimization might not have any incentive to follow a
            "human-sensible" way to optimize. For example, provided that the system is not under tight constraint and
            has to charge 10 kWh in a battery in the next 6 hours, then the optimization can do whatever it likes (
            2.5 in first hour, then 0 then 4, etc.) as it will not change the overall cost. This was criticized in
            the OMNe project as it is then difficult for a person lambda to explain why the optimizer is behaving so.
            As a result, we use this flag to penalize the sum of the jumps in the total fleet power.

            Implementation: We had to find an additional way than using a penalty on having high power values as it
            is already taken and we wanted something on top of that. So we will now penalize the sum of the jumps (
            absolute difference between t+1 and t of the sum_power_fleet).
        :return:
        """
        # Initializing Increasing and Decreasing variables
        self.sum_power_fleet_diff = {
            t: {
                "increasing": self.model.NumVar(
                    0, self.model.infinity(), name=f"diff_greater_{t}"
                ),
                "decreasing": self.model.NumVar(
                    0, self.model.infinity(), name=f"diff_smaller_{t}"
                ),
            }
            for t in range(self.n_t - 1)
        }

        # Computing the differences
        fleet_power_before = [
            self.model.Sum(
                [
                    self.fleet_power[battery.id]["charge"][t]
                    - self.fleet_power[battery.id]["discharge"][t]
                    for battery in self.batteries
                ]
            )
            for t in range(self.n_t - 1)
        ]
        fleet_power_after = [
            self.model.Sum(
                [
                    self.fleet_power[battery.id]["charge"][t + 1]
                    - self.fleet_power[battery.id]["discharge"][t + 1]
                    for battery in self.batteries
                ]
            )
            for t in range(self.n_t - 1)
        ]
        fleet_power_diff = np.array(fleet_power_after) - np.array(fleet_power_before)

        # Adding constraint that Increasing+Decreasing == Diff at all t
        for t in range(self.n_t - 1):
            self.model.Add(
                fleet_power_diff[t]
                == self.sum_power_fleet_diff[t]["decreasing"]
                - self.sum_power_fleet_diff[t]["increasing"],
                f"fleet_power_diff_constraint_{t}",
            )

        # Costs is the sum of Increasing and Decreasing which will be minimized
        spiky_costs = np.array(
            [
                self.sum_power_fleet_diff[t]["decreasing"]
                + self.sum_power_fleet_diff[t]["increasing"]
                for t in range(self.n_t - 1)
            ]
        )
        self.cost["Spiky_Behaviour_Penalty"] = (
            self.spiky_behaviour_penalty * spiky_costs
        )
        return 0

    def _calc_battery_costs(self):
        """
        Calculate the ageing cost of using the battery.

        Returns
        -------

        """
        if sum([bat.cycle_cost_per_kwh for bat in self.batteries]) == 0:
            print(
                "WARNING: Trying to include battery cycle costs but the cycle costs are 0.Please use "
                "Battery.add_cycle_costs()"
            )
            logger.warning(
                "Trying to include battery cycle costs but the cycle costs are 0. Please use "
                "Battery.add_cycle_costs()"
            )

        self.battery_costs = {
            bat.id: self.model.NumVar(
                0, self.model.infinity(), name=f"BatteryCost_{bat.id}"
            )
            for bat in self.batteries
        }

        for bat in self.batteries:
            i = bat.id
            logger.info(
                f"Battery {i}: Adding cycle costs of {bat.cycle_cost_per_kwh} €/kWh"
            )
            print(f"Battery {i}: Adding cycle costs of {bat.cycle_cost_per_kwh} €/kWh")
            self.model.Add(
                self.battery_costs[i]
                == self.model.Sum(
                    [
                        self.fleet_power[i]["charge"][t]
                        + self.fleet_power[i]["discharge"][t]
                        for t in range(self.n_t)
                    ]
                )
                * self.dt
                * bat.cycle_cost_per_kwh
            )

        self.cost["battery_costs"] = [
            self.battery_costs[i] for i in self.battery_costs.keys()
        ]

    # ---------------------------------------------------
    #  Cost Components Price, marketed volumes (UK)
    # ---------------------------------------------------

    def _calc_cost_triad(self):
        """
        Cost component that is specific to the UK market

        Returns
        -------

        """
        # --------------------------------------------------------------------------------------------
        #  Formulation with cost variables
        # --------------------------------------------------------------------------------------------
        # # The additional variable is for tracking. If this turns out too costly we need to re-write this part
        self.cost["Triad"] = [
            self.model.NumVar(
                -self.model.infinity(), self.model.infinity(), f"cost_triad_{t}"
            )
            for t in range(self.n_t)
        ]

        # To calculate this we need to make sure we consider the site load in the right way if we want to extract
        # the actual target function from it. The baseline calculation is site load times triad prices, the calculation
        # with vehicles is in grid power. Hence we need to subtract the baseline from the optimized site to extract
        # the vehicle only part

        if self.site_load is not None:
            _ = [
                self.model.Add(
                    self.cost["Triad"][t]
                    == self.triad_tariffs_import[t]
                    * (self.grid_power["purchase"][t] - self.site_load[t])
                    * self.dt
                    - self.triad_tariffs_export[t]
                    * self.grid_power["feed"][t]
                    * self.dt,
                    f"constraint_triad_{t}",
                )
                for t in range(self.n_t)
            ]
        else:
            _ = [
                self.model.Add(
                    self.cost["Triad"][t]
                    == self.triad_tariffs_import[t]
                    * self.grid_power["purchase"][t]
                    * self.dt
                    - self.triad_tariffs_export[t]
                    * self.grid_power["feed"][t]
                    * self.dt,
                    f"constraint_triad_{t}",
                )
                for t in range(self.n_t)
            ]

        return 0

    # ---------------------------------------------------
    #  Flexibility
    # ---------------------------------------------------

    def _calc_pos_flex(self):
        """
        Calculate the positive flexibility. This is done without a specific cost/penalty in mind. It can be used
        in (at least) two distinct cases: flex forecast where we have prices for flex availability and
        flex matching where we have previously marketed flexibility that needs to be matched

        Returns
        -------

        """

        """
        The calculation of flexibility requires the minimum of three variables y_1, y_2, y_3.
        In principle this is a complicated situation in linear optimization. However, in this specific case we are 
        lucky by circumstances. The total cost function gets a correction Δy:
            y - Δy
        Now, Δy is the minimum of the above-mentioned y_i. So we need to make sure that 
            y_i >= Δy
        which is easy to program. With a lower limit for Δy of zero (which is in line with the problem) we also by 
        minimising the target function that Δy becomes as large as possible and thus hits to smallest of the y_i  
        """

        """
        Original Formulation ---------------------

        SOC[0]=(x[0]*Chargingefficiency if x[0] >=0 else x[0]/Dischargingefficiency)/2+SOC_initial

        FlexPos[0] = EV_simulation_sample['FFR Marketing'][0]*(min(Buffer_Energy_Factor*(min(SOC[0],SOC_initial)-min_energy_content),\
                                                     MaxDischargingPower + x[0]))
        for i in range(1,n):
            SOC[i]=(x[i]*Chargingefficiency if x[i] >=0 else x[i]/Dischargingefficiency)/2+SOC[i-1]
            FlexPos[i] =  EV_simulation_sample['FFR Marketing'][i]*(min(Buffer_Energy_Factor*(min(SOC[i],SOC[i-1])-min_energy_content),\
                                                     MaxDischargingPower + x[i]))  
        """
        for battery in self.batteries:
            i = battery.id

            # ---- Define Variables ----
            self.flex_pos[i] = [
                self.model.NumVar(0, self.model.infinity(), name=f"pos_flex_{i}_{t}")
                for t in range(self.n_t)
            ]

            # ---- Fill Variables with life -----
            for t in range(self.n_t):
                energy_current = self.fleet_energy[i][t]
                if t == 0:
                    energy_previous = battery.energy_start
                else:
                    energy_previous = self.fleet_energy[i][t - 1]

                # Current Power in our convention
                # Here we need the power wrt to the grid rather than the charger -> Correct for efficiency
                power_current = (
                    self.fleet_power[i]["charge"][t] / battery.efficiency_charge
                    - self.fleet_power[i]["discharge"][t] * battery.efficiency_discharge
                )

                # ---- Positive Flex -----
                # 1) Energy that could maximally be discharged:
                delta_energy_discharge_max_0 = energy_previous - battery.energy_min
                delta_energy_discharge_max_1 = energy_current - battery.energy_min

                # 2) Power from this
                # Efficiency: we get less power at the charger -> multiply
                power_discharge_max_0 = (
                    delta_energy_discharge_max_0
                    * battery.efficiency_discharge
                    / self.dt
                )
                power_discharge_max_1 = (
                    delta_energy_discharge_max_1
                    * battery.efficiency_discharge
                    / self.dt
                )

                # 3) Difference between max discharge and current charging power
                power_discharge_max_2 = (
                    battery.power_discharge_max * battery.efficiency_discharge
                    + power_current
                )

                self.model.Add(
                    self.flex_buffer * power_discharge_max_0 >= self.flex_pos[i][t],
                    f"constraint_flex_pos_1_{i}_{t}",
                )
                self.model.Add(
                    self.flex_buffer * power_discharge_max_1 >= self.flex_pos[i][t],
                    f"constraint_flex_pos_2_{i}_{t}",
                )
                self.model.Add(
                    power_discharge_max_2 >= self.flex_pos[i][t],
                    f"constraint_flex_pos_3_{i}_{t}",
                )

        # Sum flex over all batteries
        self.flex_pos_total = {
            t: self.model.NumVar(
                -self.model.infinity(),
                self.model.infinity(),
                name=f"total_pos_flex_{t}",
            )
            for t in range(self.n_t)
        }

        for t in range(self.n_t):
            # TODO: add grid efficiencies if different than 1 (as in Ladepark Duisburg)
            self.model.Add(
                self.flex_pos_total[t]
                == sum([self.flex_pos[battery.id][t] for battery in self.batteries]),
                f"constraint_flex_pos_{t}",
            )

        return 0

    def _calc_neg_flex(self):
        """
        Calculate the flexibility. This is done without a specific cost/penalty in mind. It can be used
        in (at least) two distinct cases: flex forecast where we have prices for flex availability and
        flex matching where we have previously marketed flexibility that needs to be matched

        Returns
        -------

        """

        """
        The calculation of flexibility requires the minimum of three variables y_1, y_2, y_3.
        In principle this is a complicated situation in linear optimization. However, in this specific case we are 
        lucky by circumstances. The total cost function gets a correction Δy:
            y - Δy
        Now, Δy is the minimum of the above-mentioned y_i. So we need to make sure that 
            y_i >= Δy
        which is easy to program. With a lower limit for Δy of zero (which is in line with the problem) we also by 
        minimising the target function that Δy becomes as large as possible and thus hits to smallest of the y_i  
        """

        """
        Original Formulation ---------------------
        
        SOC[0]=(x[0]*Chargingefficiency if x[0] >=0 else x[0]/Dischargingefficiency)/2+SOC_initial
                
        FlexNeg[0] = EV_simulation_sample['FFR Marketing'][0]*(min(Buffer_Energy_Factor*(max_energy_content-max(SOC[0],SOC_initial)), \
                                                    MaxChargingPower/Chargingefficiency - x[0]))
        for i in range(1,n):
            SOC[i]=(x[i]*Chargingefficiency if x[i] >=0 else x[i]/Dischargingefficiency)/2+SOC[i-1]
            FlexNeg[i] = EV_simulation_sample['FFR Marketing'][i]*(min(Buffer_Energy_Factor*(max_energy_content-max(SOC[i],SOC[i-1])), \
                                                    MaxChargingPower/Chargingefficiency - x[i]))   
        """
        for battery in self.batteries:
            i = battery.id

            # ---- Define Variables ----
            self.flex_neg[i] = [
                self.model.NumVar(0, self.model.infinity(), name=f"neg_flex_{i}_{t}")
                for t in range(self.n_t)
            ]

            # ---- Fill Variables with life -----
            for t in range(self.n_t):
                energy_current = self.fleet_energy[i][t]
                if t == 0:
                    energy_previous = battery.energy_start
                else:
                    energy_previous = self.fleet_energy[i][t - 1]

                # Current Power in our convention
                # Here we need the power wrt to the grid rather than the charger -> Correct for efficiency
                power_current = (
                    self.fleet_power[i]["charge"][t] / battery.efficiency_charge
                    - self.fleet_power[i]["discharge"][t] * battery.efficiency_discharge
                )

                # ---- Negative Flex -----

                # 1) Energy that could maximally be charged:
                delta_energy_charge_max_0 = battery.energy_max - energy_previous
                delta_energy_charge_max_1 = battery.energy_max - energy_current

                # 2) Power from this
                # Efficiency: we need power at the charger -> divide
                power_charge_max_0 = (
                    delta_energy_charge_max_0 / battery.efficiency_charge / self.dt
                )
                power_charge_max_1 = (
                    delta_energy_charge_max_1 / battery.efficiency_charge / self.dt
                )

                # 3) Difference between max charging power and current charging power
                power_charge_max_2 = (
                    battery.power_charge_max / battery.efficiency_charge - power_current
                )

                self.model.Add(
                    self.flex_buffer * power_charge_max_0 >= self.flex_neg[i][t],
                    f"constraint_flex_neg_1_{i}_{t}",
                )
                self.model.Add(
                    self.flex_buffer * power_charge_max_1 >= self.flex_neg[i][t],
                    f"constraint_flex_neg_2_{i}_{t}",
                )
                self.model.Add(
                    power_charge_max_2 >= self.flex_neg[i][t],
                    f"constraint_flex_neg_3_{i}_{t}",
                )

        # Sum flex over all batteries
        self.flex_neg_total = {
            t: self.model.NumVar(
                -self.model.infinity(),
                self.model.infinity(),
                name=f"total_neg_flex_{t}",
            )
            for t in range(self.n_t)
        }

        for t in range(self.n_t):
            # TODO: add grid efficiencies if different than 1 (as in Ladepark Duisburg)
            self.model.Add(
                self.flex_neg_total[t]
                == sum([self.flex_neg[battery.id][t] for battery in self.batteries]),
                f"constraint_flex_neg_{t}",
            )

        return 0

    def _calc_cost_pos_flex_forecast(self):
        """
        This is the OR specific part of the flex FORECAST

        Returns
        -------

        """

        # We need to compare a total of three quantities and choose the minimum.
        # To do so we calculate all three and define the variable flex_min which is <= to all three

        # Calculate the general flexibility (self.flex_pos, self.flex_neg)
        self._calc_pos_flex()

        # Cost through prices, sum over time
        self.cost["FlexPos"] = [
            -self.flex_pos_total[t] * self.prices_flex_pos[t] * self.mask_flex_pos[t]
            for t in range(self.n_t)
        ]
        return 0

    def _calc_cost_neg_flex_forecast(self):
        """
        This is the OR specific part of the flex FORECAST

        Returns
        -------

        """

        # We need to compare a total of three quantities and choose the minimum.
        # To do so we calculate all three and define the variable flex_min which is <= to all three

        # Calculate the general flexibility (self.flex_pos, self.flex_neg)
        self._calc_neg_flex()

        # Cost through prices, sum over time
        self.cost["FlexNeg"] = [
            -self.flex_neg_total[t] * self.prices_flex_neg[t] * self.mask_flex_neg[t]
            for t in range(self.n_t)
        ]

        return 0

    def add_symmetrical_flex_constraints(self):
        """
        Add constraint on positive and negative flexibility variables to be equal at every timestep.
        """
        for t in range(self.n_t):
            self.model.Add(
                self.flex_pos_total[t] == self.flex_neg_total[t],
                f"constraint_symmetrical_flex_{t}",
            )

    def _calc_cost_flex_matching(self):
        """
        This is the OR specific part of the flex MATCHING

        Returns
        -------

        """

        # Matching here is different from Marketed Volumes in N2EX. These need to be matched exactly.
        # Flexibility matching is different, here we need to match marketed flex OR MORE.

        # 1) Positive Flexibility
        if self.marketed_flex_pos is not None:
            # self._calc_pos_flex() This was already called by calc_cost_pos_flex_forecast and cannot be called twice
            flex_pos_diff = [0] * self.n_t
            flex_pos_diff_pos = [
                self.model.NumVar(
                    0, self.model.infinity(), name=f"Flex_Pos_Diff_Pos_{t}"
                )
                for t in range(self.n_t)
            ]
            flex_pos_diff_neg = [
                self.model.NumVar(
                    0, self.model.infinity(), name=f"Flex_Pos_Diff_Neg_{t}"
                )
                for t in range(self.n_t)
            ]
            for t in range(self.n_t):
                if self.mask_flex_pos[t]:
                    # Both are in kW
                    flex_pos_diff[t] = (
                        self.flex_pos_total[t] - self.marketed_flex_pos[t]
                    )
                else:
                    flex_pos_diff[t] = 0
                self.model.Add(
                    flex_pos_diff[t] == flex_pos_diff_pos[t] - flex_pos_diff_neg[t],
                    f"Flex_Pos_Diff_{t}",
                )

            # Cost through prices, sum over time
            self.cost["FlexMatchPos"] = [flex_pos_diff_neg[t] for t in range(self.n_t)]

        # 2) Negative Flexibility
        if self.marketed_flex_neg is not None:
            # self._calc_neg_flex() This was already called by calc_cost_pos_flex_forecast and cannot be called twice

            flex_neg_diff_pos = [
                self.model.NumVar(
                    0, self.model.infinity(), name=f"Flex_Neg_Diff_Pos_{t}"
                )
                for t in range(self.n_t)
            ]
            flex_neg_diff_neg = [
                self.model.NumVar(
                    0, self.model.infinity(), name=f"Flex_Neg_Diff_Neg_{t}"
                )
                for t in range(self.n_t)
            ]

            flex_neg_diff = [0] * self.n_t
            for t in range(self.n_t):
                if self.mask_flex_neg[t]:
                    # Both are in kW
                    flex_neg_diff[t] = (
                        self.flex_neg_total[t] - self.marketed_flex_neg[t]
                    )
                else:
                    flex_neg_diff[t] = 0
                self.model.Add(
                    flex_neg_diff[t] == flex_neg_diff_pos[t] - flex_neg_diff_neg[t],
                    f"Flex_Neg_Diff_{t}",
                )

            # Cost through prices, sum over time
            self.cost["FlexMatchNeg"] = [flex_neg_diff_neg[t] for t in range(self.n_t)]

        return 0

    # ---------------------------------------------------
    #  Main Optimizer
    # ---------------------------------------------------

    def _calc_objectives(self):
        """
        For readability all these checks are combined into one method

        Returns
        -------

        """

        # Tariffs, Flex and Marketed Volumes may or may not contribute to the cost function
        if (self.tariffs_export is not None) or (self.tariffs_import is not None):
            self._calc_cost_prices()

        if (self.triad_tariffs_export is not None) or (
            self.triad_tariffs_import is not None
        ):
            self._calc_cost_triad()

        if (self.capacity_tariffs_export is not None) or (
            self.capacity_tariffs_import is not None
        ):
            self._calc_capacity_prices()

        if self.marketed_volumes is not None:
            self._calc_cost_marketed_volumes()

        # Calculate cost from flex FORECAST?
        if self.prices_flex_pos is not None:
            self._calc_cost_pos_flex_forecast()
        if self.prices_flex_neg is not None:
            self._calc_cost_neg_flex_forecast()
        if self.symmetrical_flex:
            self.add_symmetrical_flex_constraints()

        # Flex matching?
        if (self.marketed_flex_pos is not None) or (self.marketed_flex_neg is not None):
            self._calc_cost_flex_matching()

        # Site Limits
        if (self.site_load_restriction_charge is not None) or (
            self.site_load_restriction_discharge is not None
        ):
            self._calc_site_limits()

        # Do we penalize charging? This is designed to prolong the battery lifetime
        if self.include_battery_costs:
            self._calc_battery_costs()

        # Do we penalize spiky behaviour? This is designed to make the results look nice to humans
        if self.penalize_spiky_behaviour:
            self._calc_cost_spiky_behaviour()

        return 0

    def optimize(self):
        # ------------------------------------------------
        #  Initialize
        # ------------------------------------------------
        self.model = pywraplp.Solver(
            "model", pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING
        )

        self.model.Clear()
        self.model.SuppressOutput()

        # Initialize Basic Parameters
        self._initialize_grid_power()

        # [OPTIONAL]
        if self.charging_points:
            self.assign_batteries_to_charging_point()

        _ = [self._initialize_battery(battery) for battery in self.batteries]

        self._initialize_fleet_power()

        # Calculate the objective function parts depending on the setup
        self._calc_objectives()

        # ------------------------------------------------
        #  Optimizer
        # ------------------------------------------------
        logger.debug(
            f"Model: constraints = {self.model.NumConstraints()} | variables = {self.model.NumVariables()} \n"
            + f"time steps: {self.n_t} vehicles: {len(self.batteries)}"
        )
        logger.debug(f"costs: {self.cost.keys()}")

        self.model.Minimize(sum([sum(v) for v in self.cost.values()]))

        start_solving = pd.Timestamp.now()

        # self.model.EnableOutput()
        self.status = self.model.Solve()

        self.solve_time = pd.Timedelta(
            pd.Timestamp.now() - start_solving
        ).total_seconds()
        self.objective_value = (
            self.model.Objective().Value() if self.status == 0 else np.nan
        )

        if self.status == 0:
            logger.debug(
                f"Problem solved | {self.model.wall_time()} milliseconds | "
                f"{self.model.iterations()} iterations"
            )
        else:
            logger.warning("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            logger.warning("Not solved! Assign linear charging")
            logger.warning("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        # ------------------------------------------------
        #  Parse Result
        # ------------------------------------------------
        result = SiteResult.create(
            self.__dict__, optimizer="or", default_strategy=self.default_strategy
        )

        return result


def main():
    x = np.linspace(0, 2 * np.pi, 30)
    prices_import = np.sin(x) + 2

    battery1 = Battery(
        **{
            "id": 42,
            "capacity": 40,
            "energy_start": 10,
            "energy_end": 40,
            "energy_min": 5,
            "energy_max": 40,
            "charge_max": 5,
            "discharge_max": 0,
            "connected": [False] * 7 + [True] * 18 + [False] * 5,
        }
    )

    battery2 = Battery(
        **{
            "id": 23,
            "capacity": 40,
            "energy_start": 12,
            "energy_end": 40,
            "energy_min": 5,
            "energy_max": 40,
            "charge_max": 5,
            "discharge_max": 0,
            "connected": [False] * 5 + [True] * 20 + [False] * 5,
        }
    )

    battery3 = Battery(
        **{
            "id": 3,
            "capacity": 40,
            "energy_start": 12,
            "energy_end": 40,
            "energy_min": 5,
            "energy_max": 40,
            "charge_max": 5,
            "discharge_max": 0,
            "connected": [False] * 7 + [True] * 16 + [False] * 7,
        }
    )

    fo = FleetOptimizationOR(id=1, calculate_savings=True)
    _ = [fo.add_battery(battery) for battery in [battery1, battery2, battery3]]

    #
    fo.add_prices(
        tariffs_import=prices_import,
        tariffs_export=prices_import,
        capacity_tariffs_import=10,
    )

    # fo.add_marketed_volumes(np.array([-1] * 5 + [np.nan] * 25))

    # Flexibility Product
    # price_flex_pos = [0] * 13 + [1] * 4 + [0] * 13
    # fo.add_flex(prices_flex_pos=price_flex_pos)

    fo.add_site_limits(
        site_load_restriction_discharge=0, site_load_restriction_charge=20
    )

    fo.add_site_load(np.abs(np.cos(x) * 17))

    fo.late_charging = 7

    res = fo.optimize()

    pathfig = "../figures"
    os.makedirs(pathfig, exist_ok=True)
    fo.plot(res, f"{pathfig}/OR2.png")


if __name__ == "__main__":
    main()
