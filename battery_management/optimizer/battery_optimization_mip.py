import numpy as np
import pandas as pd
from battery_management.assets.battery import Battery
from battery_management.helper.site_result import SiteResult
from battery_management.optimizer.battery_optimization_baseclass import (
    FleetOptimizationBaseclass,
)
from mip import BINARY, CONTINUOUS, Model, minimize, xsum

np.random.seed(42)


class FleetOptimizationMIP(FleetOptimizationBaseclass):
    def __init__(self, id: int, calculate_savings: bool = False):
        """
        Initialize the Fleet Optimization MIP.

        Parameters
        ----------
        id : int
            Identifier for the optimization instance.
        calculate_savings : bool, optional
            Flag to determine if savings should be calculated, by default False.
        """
        super().__init__(id, calculate_savings)

        self.grid_power = {"feed": [], "purchase": []}
        self.fleet_power = {}
        self.fleet_energy = {}

        self.grid_bool = {"feed": [], "purchase": []}
        self.fleet_bool = {}
        self.sum_power_fleet = {"charge": {}, "discharge": {}}

        self.flex_pos = {}
        self.flex_neg = {}

        self.model = None
        self.cost = {}

    def _initialize_grid_power(self):
        """Initialize grid power variables and constraints."""
        for action in ["feed", "purchase"]:
            self.grid_power[action] = [
                self.model.add_var(
                    var_type=CONTINUOUS,
                    lb=0,
                    ub=10000
                    if getattr(self, f"site_load_restriction_{action}") is None
                    else getattr(self, f"site_load_restriction_{action}"),
                    name=f"grid_power_{action}_{t}",
                )
                for t in range(self.n_t)
            ]
            self.grid_bool[action] = [
                self.model.add_var(var_type=BINARY, name=f"grid_bool_{action}_{t}")
                for t in range(self.n_t)
            ]

        for t in range(self.n_t):
            self.model.add_constr(
                self.grid_bool["feed"][t] + self.grid_bool["purchase"][t] == 1,
                f"prevent_simultaneous_feed_in_and_purchase_{t}",
            )

    def _initialize_fleet_power(self):
        """Initialize fleet power variables and constraints."""
        constraints = [
            self.model.constr_by_name(f"overall_electricity_balance_{t}")
            for t in range(self.n_t)
            if self.model.constr_by_name(f"overall_electricity_balance_{t}") is not None
        ]
        if constraints:
            self.model.remove(constraints)

        for t in range(self.n_t):
            for action in ["charge", "discharge"]:
                self.sum_power_fleet[action][t] = xsum(
                    [
                        self.fleet_power[battery.id][action][t]
                        for battery in self.batteries
                    ]
                )

            source = (
                self.grid_power["purchase"][t] + self.sum_power_fleet["discharge"][t]
            )
            sink = self.grid_power["feed"][t] + self.sum_power_fleet["charge"][t]
            if self.site_load is not None:
                sink += self.site_load[t]
            self.model += (source == sink, f"overall_electricity_balance_{t}")

    def _initialize_battery(self, battery: Battery):
        """
        Initialize battery variables and constraints.

        Parameters
        ----------
        battery : Battery
            Battery instance to initialize.
        """
        i = battery.id
        for action in ["charge", "discharge"]:
            self.fleet_power[i][action] = [
                self.model.add_var(
                    var_type=CONTINUOUS,
                    lb=0,
                    ub=(
                        getattr(battery, f"{action}_max") if battery.connected[t] else 0
                    ),
                    name=f"power_{i}_{action}_{t}",
                )
                for t in range(self.n_t)
            ]
            self.fleet_bool[i][action] = [
                self.model.add_var(var_type=BINARY, name=f"bool_{i}_{action}_{t}")
                for t in range(self.n_t)
            ]

        self.fleet_energy[i] = [
            self.model.add_var(
                var_type=CONTINUOUS,
                lb=battery.energy_min,
                ub=battery.energy_max,
                name=f"energy_{i}_{t}",
            )
            for t in range(self.n_t)
        ]

        self.fleet_energy[i][0] = (
            battery.energy_start
            + self.fleet_power[i]["charge"][0]
            - self.fleet_power[i]["discharge"][0]
        )

        for t in range(1, self.n_t):
            self.fleet_energy[i][t] = (
                self.fleet_energy[i][t - 1]
                + self.fleet_power[i]["charge"][t]
                - self.fleet_power[i]["discharge"][t]
            )

            self.model += (
                self.fleet_energy[i][t] >= battery.energy_min,
                f"energy_min_constraint_{i}_{t}",
            )
            self.model += (
                self.fleet_energy[i][t] <= battery.energy_max,
                f"energy_max_constraint_{i}_{t}",
            )

        self.model += (
            self.fleet_energy[i][self.n_t - 1] >= battery.energy_end,
            f"fully_charged_constraint_{i}",
        )

        for t in range(self.n_t):
            self.model += (
                self.fleet_power[i]["charge"][t]
                <= self.fleet_bool[i]["charge"][t] * battery.charge_max,
                f"maximum_limit_battery_charging_{i}_{t}",
            )
            self.model += (
                self.fleet_power[i]["discharge"][t]
                <= self.fleet_bool[i]["discharge"][t] * battery.discharge_max,
                f"maximum_limit_battery_discharging_{i}_{t}",
            )
            self.model += (
                self.fleet_bool[i]["charge"][t] + self.fleet_bool[i]["discharge"][t]
                <= 1,
                f"prevent_simultaneous_charging_and_discharging_{i}_{t}",
            )

    def _calc_cost_prices(self):
        """Calculate cost prices."""
        cost_spot = [
            self.model.add_var(
                var_type=CONTINUOUS, lb=-10000, ub=10000, name=f"cost_spot_{t}"
            )
            for t in range(self.n_t)
        ]

        mm = (
            self.mask_marketed if self.mask_marketed is not None else np.zeros(self.n_t)
        )

        for t in range(self.n_t):
            self.model += (
                cost_spot[t]
                == (1 - mm[t])
                * (
                    self.tariffs_import[t] * self.sum_power_fleet["charge"][t]
                    - self.tariffs_export[t] * self.sum_power_fleet["discharge"][t]
                ),
                f"constr_cost_spot_{t}",
            )

        self.cost["Spot"] = cost_spot

    def _calc_cost_marketed_volumes(self):
        """Calculate cost for marketed volumes."""
        constraints = [
            self.model.constr_by_name(f"marketed_volumes_diff_{t}")
            for t in range(self.n_t)
            if self.model.constr_by_name(f"marketed_volumes_diff_{t}") is not None
        ]
        if constraints:
            self.model.remove(constraints)

        diff_pos = [
            self.model.add_var(
                var_type=CONTINUOUS,
                lb=0,
                ub=10000,
                name=f"marketed_volumes_Diff_Pos_{t}",
            )
            for t in range(self.n_t)
        ]
        diff_neg = [
            self.model.add_var(
                var_type=CONTINUOUS,
                lb=0,
                ub=10000,
                name=f"marketed_volumes_Diff_neg_{t}",
            )
            for t in range(self.n_t)
        ]

        diff = [0] * self.n_t
        for t in range(self.n_t):
            if self.mask_marketed[t]:
                diff[t] = (
                    self.marketed_volumes[t]
                    - self.sum_power_fleet["charge"][t]
                    + self.sum_power_fleet["discharge"][t]
                )
            self.model += (
                diff[t] == diff_pos[t] - diff_neg[t],
                f"marketed_volumes_diff_{t}",
            )

        cost_marketed = [
            self.model.add_var(
                var_type=CONTINUOUS, lb=0, ub=10000, name=f"cost_marketed_{t}"
            )
            for t in range(self.n_t)
        ]

        for t in range(self.n_t):
            self.model += (
                cost_marketed[t] == 10 * (diff_pos[t] + diff_neg[t]),
                f"constr_cost_marketed_{t}",
            )

        self.cost["MarketedVolumes"] = cost_marketed

    def _calc_cost_flex(self):
        """Calculate cost for flexibility."""
        for battery in self.batteries:
            i = battery.id

            self.flex_pos[i] = [
                self.model.add_var(
                    var_type=CONTINUOUS,
                    lb=0,
                    ub=battery.discharge_max + battery.charge_max,
                    name=f"pos_flex_{i}_{t}",
                )
                for t in range(self.n_t)
            ]
            self.flex_neg[i] = [
                self.model.add_var(
                    var_type=CONTINUOUS,
                    lb=0,
                    ub=battery.discharge_max + battery.charge_max,
                    name=f"neg_flex_{i}_{t}",
                )
                for t in range(self.n_t)
            ]

            for t in range(self.n_t):
                self.model += (
                    self.flex_buffer * self.fleet_energy[i][t] - battery.energy_min
                    >= self.flex_pos[i][t],
                    f"flex_pos_constraint_1_{i}_{t}",
                )
                self.model += (
                    self.flex_buffer
                    * (
                        self.fleet_energy[i][t]
                        - self.fleet_power[i]["charge"][t]
                        + self.fleet_power[i]["discharge"][t]
                    )
                    - battery.energy_min
                    >= self.flex_pos[i][t],
                    f"flex_pos_constraint_2_{i}_{t}",
                )
                self.model += (
                    battery.discharge_max
                    + self.fleet_power[i]["charge"][t]
                    - self.fleet_power[i]["discharge"][t]
                    >= self.flex_pos[i][t],
                    f"flex_pos_constraint_3_{i}_{t}",
                )

        self.cost["FlexPos"] = [
            self.model.add_var(
                var_type=CONTINUOUS, lb=-10000, ub=0, name=f"pos_total_flex_{t}"
            )
            for t in range(self.n_t)
        ]

        for t in range(self.n_t):
            self.model += (
                self.cost["FlexPos"][t]
                == -self.prices_flex_pos[t]
                * xsum([self.flex_pos[battery.id][t] for battery in self.batteries]),
                f"cost_pos_flex_{t}",
            )

    def info(self):
        """Print model variables and constraints."""
        print("-" * 125)
        for i, x in enumerate(self.model.constrs):
            print(i, x)
        print("-" * 125)
        for i, x in enumerate(self.model.vars):
            print(i, x)
        print("-" * 125)

    def optimize(self) -> SiteResult:
        """
        Execute the MIP Optimizer.

        Returns
        -------
        Result
            Optimization result object.
        """
        self.model = Model("BatteryOptimizer")
        self.model.verbose = 0
        self._initialize_grid_power()
        for battery in self.batteries:
            self._initialize_battery(battery)
        self._initialize_fleet_power()

        if self.tariffs_import is not None or self.tariffs_export is not None:
            self._calc_cost_prices()

        if self.marketed_volumes is not None:
            self._calc_cost_marketed_volumes()

        start = pd.Timestamp.now()
        self.model.objective = minimize(xsum([xsum(v) for v in self.cost.values()]))
        self.model.optimize(max_seconds=300)

        time_elapsed = pd.Timedelta(pd.Timestamp.now() - start).total_seconds()

        cost = {k: [x.x for x in v] for k, v in self.cost.items()}
        cost_total = sum([sum(v) for v in cost.values()])

        energy_deltas = {
            battery.id: [
                self.fleet_power[battery.id]["charge"][t].x
                - self.fleet_power[battery.id]["discharge"][t].x
                for t in range(self.n_t)
            ]
            for battery in self.batteries
        }

        df = self._calculate_result_df(energy_deltas)

        for k, v in cost.items():
            df["site"][f"Objective_{k}"] = v

        return SiteResult(
            df, objective_value=cost_total, time_elapsed=time_elapsed, success=0
        )


if __name__ == "__main__":
    x = np.linspace(0, 2 * np.pi, 30)
    prices_import = np.sin(x)

    battery1 = Battery(
        id=42,
        capacity=40,
        energy_start=10,
        energy_end=40,
        energy_min=5,
        energy_max=40,
        charge_max=5,
        discharge_max=5,
        connected=[True] * 30,
    )

    battery2 = Battery(
        id=23,
        capacity=40,
        energy_start=12,
        energy_end=40,
        energy_min=5,
        energy_max=40,
        charge_max=5,
        discharge_max=5,
        connected=[False] * 5 + [True] * 25,
    )

    fo = FleetOptimizationMIP(id=1)
    for battery in [battery1, battery2]:
        fo.add_battery(battery)
    fo.add_prices(prices_import, prices_import)

    marketed_volumes = np.array([2] * 5 + [np.nan] * 25)
    fo.add_marketed_volumes(marketed_volumes)

    fo.add_site_load(np.array([1] * 30))

    fo.add_site_limits(site_load_restriction_discharge=0)
    result = fo.optimize()
    fo.plot(result.df, "../figures/mip_restriction.png")
    print(result.df)
