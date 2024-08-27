import os
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from battery_management.assets.battery import Battery
from battery_management.helper.site_result import SiteResult
from battery_management.optimizer.battery_optimization_baseclass import (
    FleetOptimizationBaseclass,
)
from loguru import logger
from ortools.linear_solver import pywraplp

np.random.seed(42)


class FleetOptimizationOR(FleetOptimizationBaseclass):
    """
    ORTools implementation of the FleetOptimization.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the FleetOptimizationOR class.

        Parameters
        ----------
        id : int
            The ID used when handling more than one optimization at a time.
        dt : float
            Time discretization to convert power into energy.
            If dt=1, then it can be seen as using energy kWh (v2g case).
            If using power and time interval is 15min (eva case) then dt should be 0.25.
        *args
        **kwargs
        """
        super().__init__(**kwargs)

        self.type = "OR"
        self.id_list = set()
        self.batteries: List[Battery] = []

        self.grid_power: Dict[str, Dict[int, pywraplp.Variable]] = {
            "feed": {},
            "purchase": {},
            "curtail": {},
        }
        self.grid_peak: Dict[str, Optional[pywraplp.Variable]] = {
            "feed": None,
            "purchase": None,
        }
        self.fleet_power: Dict[int, Dict[str, Dict[int, pywraplp.Variable]]] = {}
        self.fleet_energy: Dict[int, Dict[int, pywraplp.Variable]] = {}

        self.grid_bool: Dict[str, Dict[int, pywraplp.Variable]] = {
            "feed": {},
            "purchase": {},
            "curtail": {},
        }
        self.fleet_bool: Dict[int, Dict[str, Dict[int, pywraplp.Variable]]] = {}
        self.sum_power_fleet: Dict[str, Dict[int, pywraplp.Variable]] = {
            "charge": {},
            "discharge": {},
        }

        self.site_constraint: Dict[str, Optional[pywraplp.Variable]] = {
            "feed": None,
            "purchase": None,
        }

        self.flex_pos: Dict[int, List[pywraplp.Variable]] = {}
        self.flex_neg: Dict[int, List[pywraplp.Variable]] = {}

        self.cost: Dict[str, List[Union[pywraplp.Variable, float]]] = {}

        self.limit_purchase_cost: Optional[float] = None
        self.limit_purchase_penalty: float = 5000
        self.fully_charged_penalty: float = 1000
        self.spiky_behaviour_penalty: float = 0.01

        self.model: Optional[pywraplp.Solver] = None
        self.results: Dict = {}

    def _initialize_grid_power(self) -> None:
        """
        Create the grid power variables for each time step.
        """
        for t in range(self.n_t):
            for key in self.grid_power:
                label = f"grid_power_{key}_{t}"
                upper_limit = (
                    0
                    if not self.allow_curtailment and key == "curtail"
                    else self.model.infinity()
                )
                self.grid_power[key][t] = self.model.NumVar(0, upper_limit, label)

            for key in self.grid_bool:
                label = f"grid_bool_{key}_{t}"
                self.grid_bool[key][t] = self.model.BoolVar(label)

            if not self.allow_curtailment:
                self.model.Add(
                    self.grid_bool["curtail"][t] == 0, f"disable_curtailment_{t}"
                )

    def _initialize_battery(self, battery: Battery) -> Dict[str, str]:
        """
        Initialize the power and energy variables for individual batteries and chargers.

        Parameters
        ----------
        battery : Battery
            The battery to initialize.

        Returns
        -------
        Dict[str, str]
            A dictionary containing a message if the battery could not be initialized.
        """
        if not battery.is_connected():
            return {"message": f"Battery {battery.id} is not connected."}

        i = battery.id

        self.fleet_power[i] = {"charge": {}, "discharge": {}}
        self.fleet_energy[i] = {}
        self.fleet_bool[i] = {"charge": {}, "discharge": {}, "disconnected": {}}

        for t in range(self.n_t):
            self._initialize_battery_variables(battery, t)

        self._initialize_battery_energy(battery)

        if not battery.stationary:
            self._handle_non_stationary_battery(battery)

        return {}

    def _initialize_battery_variables(self, battery: Battery, t: int) -> None:
        """
        Initialize variables for a specific battery at a given time step.

        Parameters
        ----------
        battery : Battery
            The battery to initialize variables for.
        t : int
            The time step.
        """
        i = battery.id
        connected = battery.connected[t]

        self.fleet_power[i]["charge"][t] = self.model.NumVar(
            0, battery.power_charge_max if connected else 0, f"power_{i}_charge_{t}"
        )
        self.fleet_power[i]["discharge"][t] = self.model.NumVar(
            0,
            battery.power_discharge_max if connected else 0,
            f"power_{i}_discharge_{t}",
        )
        self.fleet_energy[i][t] = self.model.NumVar(
            battery.energy_min, battery.energy_max, f"energy_{i}_{t}"
        )

        self.fleet_bool[i]["charge"][t] = self.model.BoolVar(f"bool_{i}_charge_{t}")
        self.fleet_bool[i]["discharge"][t] = self.model.BoolVar(
            f"bool_{i}_discharge_{t}"
        )
        self.fleet_bool[i]["disconnected"][t] = self.model.BoolVar(
            f"bool_{i}_disconnected_{t}"
        )

    def _initialize_battery_energy(self, battery: Battery) -> None:
        """
        Initialize energy constraints for a battery.

        Parameters
        ----------
        battery : Battery
            The battery to initialize energy constraints for.
        """
        i = battery.id

        self.fleet_energy[i][0] = (
            battery.energy_start
            + self.fleet_power[i]["charge"][0] * self.dt
            - self.fleet_power[i]["discharge"][0] * self.dt
        )
        self.model.Add(
            self.fleet_energy[i][0] >= battery.energy_min,
            f"energy_min_constraint_{i}_0",
        )
        self.model.Add(
            self.fleet_energy[i][0] <= battery.energy_max,
            f"energy_max_constraint_{i}_0",
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

    def _handle_non_stationary_battery(self, battery: Battery) -> None:
        """
        Handle constraints for non-stationary batteries (EVs).

        Parameters
        ----------
        battery : Battery
            The non-stationary battery to handle.
        """
        i = battery.id

        if not self.fully_charged_as_penalty:
            self.model.Add(
                self.fleet_energy[i][self.n_t - 1] >= battery.energy_end,
                f"fully_charged_constraint_{i}",
            )
        else:
            remaining_energy_not_charged = (
                battery.energy_end - self.fleet_energy[i][self.n_t - 1]
            )

            if "fully_charged_penalty" not in self.cost:
                self.cost["fully_charged_penalty"] = []

            self.cost["fully_charged_penalty"].append(
                remaining_energy_not_charged * self.fully_charged_penalty
            )

    def _initialize_fleet_power(self) -> None:
        """
        Initialize overall electricity balance and constraints.
        """
        for t in range(self.n_t):
            self._initialize_electricity_balance(t)
            self._prevent_simultaneous_grid_operations(t)

        self._calculate_total_fleet_power()

    def _initialize_electricity_balance(self, t: int) -> None:
        """
        Initialize electricity balance for a specific time step.

        Parameters
        ----------
        t : int
            The time step.
        """
        source = self.grid_power["purchase"][t] * self.purchase_efficiency + sum(
            self.fleet_power[battery.id]["discharge"][t] * battery.efficiency_discharge
            for battery in self.batteries
        )

        sink = (
            self.grid_power["feed"][t] / self.feed_efficiency
            + self.grid_power["curtail"][t]
            + sum(
                self.fleet_power[battery.id]["charge"][t] / battery.efficiency_charge
                for battery in self.batteries
            )
        )

        if self.site_load is not None:
            sl = self.site_load[t]
            sink += sl if sl > 0 else -sl

        self.model.Add(source == sink, f"overall_electricity_balance_{t}")

    def _prevent_simultaneous_grid_operations(self, t: int) -> None:
        """
        Prevent simultaneous feed-in, curtailment, and purchase from the grid.

        Parameters
        ----------
        t : int
            The time step.
        """
        self.model.Add(
            self.grid_bool["feed"][t]
            + self.grid_bool["curtail"][t]
            + self.grid_bool["purchase"][t]
            == 1,
            f"prevent_simultaneous_feed_in_and_purchase_{t}",
        )

    def _calculate_total_fleet_power(self) -> None:
        """
        Calculate total fleet power for charging and discharging.
        """
        for t in range(self.n_t):
            self.sum_power_fleet["charge"][t] = sum(
                self.fleet_power[battery.id]["charge"][t] for battery in self.batteries
            )
            self.sum_power_fleet["discharge"][t] = sum(
                self.fleet_power[battery.id]["discharge"][t]
                for battery in self.batteries
            )

    def _calc_site_limits(self) -> None:
        """
        Calculate site limits for charging or discharging.
        """
        if self.limit_as_penalty:
            self._calc_site_limits_as_penalty()
        else:
            self._calc_site_limits_as_constraint()

    def _calc_site_limits_as_penalty(self) -> None:
        """
        Calculate site limits as a penalty in the objective function.
        """
        if self.site_load_restriction_charge is not None:
            self.site_constraint["purchase"] = self.model.NumVar(
                self.site_load_restriction_charge,
                self.model.infinity(),
                "site_constraint_purchase",
            )
            for t in range(self.n_t):
                self.model.Add(
                    self.grid_power["purchase"][t] <= self.site_constraint["purchase"],
                    f"site_import_constraint_{t}",
                )

            self.cost["site_limit_purchase"] = [
                (self.site_constraint["purchase"] - self.site_load_restriction_charge)
                * self.limit_purchase_penalty
            ]

    def _calc_site_limits_as_constraint(self) -> None:
        """
        Calculate site limits as hard constraints in the optimizer.
        """
        for t in range(self.n_t):
            if self.site_load_restriction_discharge is not None:
                self.model.Add(
                    self.grid_power["feed"][t]
                    <= self.grid_bool["feed"][t] * self.site_load_restriction_discharge,
                    f"maximum_limit_feed_in_{t}",
                )
            if self.site_load_restriction_charge is not None:
                self.model.Add(
                    self.grid_power["purchase"][t]
                    <= self.grid_bool["purchase"][t]
                    * self.site_load_restriction_charge,
                    f"maximum_limit_purchase_{t}",
                )

    def _calc_objectives(self) -> None:
        """
        Calculate all objective components based on the problem setup.
        """
        if self.tariffs_export is not None or self.tariffs_import is not None:
            self._calc_cost_prices()

        if (
            self.triad_tariffs_export is not None
            or self.triad_tariffs_import is not None
        ):
            self._calc_cost_triad()

        if (
            self.capacity_tariffs_export is not None
            or self.capacity_tariffs_import is not None
        ):
            self._calc_capacity_prices()

        if self.marketed_volumes is not None:
            self._calc_cost_marketed_volumes()

        if self.prices_flex_pos is not None:
            self._calc_cost_pos_flex_forecast()
        if self.prices_flex_neg is not None:
            self._calc_cost_neg_flex_forecast()
        if self.symmetrical_flex:
            self.add_symmetrical_flex_constraints()

        if self.marketed_flex_pos is not None or self.marketed_flex_neg is not None:
            self._calc_cost_flex_matching()

        if (
            self.site_load_restriction_charge is not None
            or self.site_load_restriction_discharge is not None
        ):
            self._calc_site_limits()

        if self.include_battery_costs:
            self._calc_battery_costs()

        if self.penalize_spiky_behaviour:
            self._calc_cost_spiky_behaviour()

    def optimize(self) -> SiteResult:
        """
        Perform the optimization process.

        Returns
        -------
        SiteResult
            The optimization result.
        """
        self.model = pywraplp.Solver(
            "model", pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING
        )

        self.model.Clear()
        self.model.SuppressOutput()

        self._initialize_grid_power()

        if self.charging_points:
            self.assign_batteries_to_charging_point()

        for battery in self.batteries:
            self._initialize_battery(battery)

        self._initialize_fleet_power()
        self._calc_objectives()

        logger.debug(
            f"Model: constraints = {self.model.NumConstraints()} | variables = {self.model.NumVariables()} \n"
            f"time steps: {self.n_t} vehicles: {len(self.batteries)}"
        )
        logger.debug(f"costs: {self.cost.keys()}")

        self.model.Minimize(sum(sum(v) for v in self.cost.values()))

        start_solving = pd.Timestamp.now()
        self.status = self.model.Solve()
        self.solve_time = (pd.Timestamp.now() - start_solving).total_seconds()
        self.objective_value = (
            self.model.Objective().Value() if self.status == 0 else np.nan
        )

        if self.status == 0:
            logger.debug(
                f"Problem solved | {self.model.wall_time()} milliseconds | "
                f"{self.model.iterations()} iterations"
            )
        else:
            logger.warning("Not solved! Assigning linear charging")

        result = SiteResult.create(
            self.__dict__, optimizer="or", default_strategy=self.default_strategy
        )

        return result

    def _calc_cost_prices(self) -> None:
        """
        Calculate the penalty from energy prices.
        """
        mm = (
            self.mask_marketed if self.mask_marketed is not None else np.zeros(self.n_t)
        )

        site_load_pos = (
            np.where(self.site_load > 0, self.site_load, 0)
            if self.site_load is not None
            else np.zeros(self.n_t)
        )
        site_load_neg = (
            np.where(self.site_load < 0, -self.site_load, 0)
            if self.site_load is not None
            else np.zeros(self.n_t)
        )

        self.cost["Spot"] = [
            self.model.NumVar(
                -self.model.infinity(), self.model.infinity(), name=f"cost_tariffs_{t}"
            )
            for t in range(self.n_t)
        ]

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
            cost_import = [0] * self.n_t

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
            cost_export = [0] * self.n_t

        for t in range(self.n_t):
            if self.tariffs_export is not None and self.tariffs_import is not None:
                self.model.Add(
                    self.cost["Spot"][t] == cost_import[t] + cost_export[t],
                    f"const_tariff_var_{t}",
                )
            elif self.tariffs_export is not None:
                self.model.Add(
                    self.cost["Spot"][t] == cost_export[t],
                    f"const_tariff_var_{t}",
                )
            elif self.tariffs_import is not None:
                self.model.Add(
                    self.cost["Spot"][t] == cost_import[t],
                    f"const_tariff_var_{t}",
                )

    def _calc_capacity_prices(self) -> None:
        """
        Calculate capacity prices for import and export.
        """
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
                self.capacity_tariffs_import * self.grid_peak["purchase"]
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
                self.capacity_tariffs_export * self.grid_peak["feed"]
            ]

    def _calc_cost_marketed_volumes(self) -> None:
        """
        Calculate the penalty for matching marketed volumes.
        """
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

        for t in range(self.n_t):
            diff = (
                self.marketed_volumes[t]
                - self.dt
                * (
                    self.sum_power_fleet["charge"][t]
                    - self.sum_power_fleet["discharge"][t]
                )
                if self.mask_marketed[t]
                else 0
            )
            self.model.Add(
                diff == diff_pos[t] - diff_neg[t], f"marketed_volumes_diff_{t}"
            )

        self.cost["MarketedVolumes"] = [
            10 * (diff_pos[t] + diff_neg[t]) for t in range(self.n_t)
        ]

    def _calc_cost_spiky_behaviour(self) -> None:
        """
        Calculate the penalty for spiky behavior in the fleet power.
        """
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

        fleet_power_before = [
            sum(
                self.fleet_power[battery.id]["charge"][t]
                - self.fleet_power[battery.id]["discharge"][t]
                for battery in self.batteries
            )
            for t in range(self.n_t - 1)
        ]
        fleet_power_after = [
            sum(
                self.fleet_power[battery.id]["charge"][t + 1]
                - self.fleet_power[battery.id]["discharge"][t + 1]
                for battery in self.batteries
            )
            for t in range(self.n_t - 1)
        ]
        fleet_power_diff = np.array(fleet_power_after) - np.array(fleet_power_before)

        for t in range(self.n_t - 1):
            self.model.Add(
                fleet_power_diff[t]
                == self.sum_power_fleet_diff[t]["decreasing"]
                - self.sum_power_fleet_diff[t]["increasing"],
                f"fleet_power_diff_constraint_{t}",
            )

        spiky_costs = [
            self.sum_power_fleet_diff[t]["decreasing"]
            + self.sum_power_fleet_diff[t]["increasing"]
            for t in range(self.n_t - 1)
        ]
        self.cost["Spiky_Behaviour_Penalty"] = [
            self.spiky_behaviour_penalty * cost for cost in spiky_costs
        ]


def main():
    x = np.linspace(0, 2 * np.pi, 30)
    prices_import = np.sin(x) + 2

    batteries = [
        Battery(
            id=42,
            capacity=40,
            energy_start=10,
            energy_end=40,
            energy_min=5,
            energy_max=40,
            charge_max=5,
            discharge_max=0,
            connected=[False] * 7 + [True] * 18 + [False] * 5,
        ),
        Battery(
            id=23,
            capacity=40,
            energy_start=12,
            energy_end=40,
            energy_min=5,
            energy_max=40,
            charge_max=5,
            discharge_max=0,
            connected=[False] * 5 + [True] * 20 + [False] * 5,
        ),
        Battery(
            id=3,
            capacity=40,
            energy_start=12,
            energy_end=40,
            energy_min=5,
            energy_max=40,
            charge_max=5,
            discharge_max=0,
            connected=[False] * 7 + [True] * 16 + [False] * 7,
        ),
    ]

    fo = FleetOptimizationOR(id=1, calculate_savings=True)
    for battery in batteries:
        fo.add_battery(battery)

    fo.add_prices(
        tariffs_import=prices_import,
        tariffs_export=prices_import,
        capacity_tariffs_import=10,
    )

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
