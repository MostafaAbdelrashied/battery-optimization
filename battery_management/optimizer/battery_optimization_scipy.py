from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from battery_management.assets.battery import Battery
from battery_management.helper.site_result import SiteResult
from battery_management.optimizer.battery_optimization_baseclass import (
    FleetOptimizationBaseclass,
)
from scipy.optimize import minimize

np.random.seed(42)


class FleetOptimizationSciPy(FleetOptimizationBaseclass):
    def __init__(
        self,
        id: int,
        calculate_savings: bool = False,
        dt: float = 1,
        penalize_charging: bool = False,
    ):
        """
        Initialize the FleetOptimizationSciPy class.

        Parameters
        ----------
        id : int
            Identifier for the optimization instance.
        calculate_savings : bool, optional
            Flag to calculate savings, by default False.
        dt : float, optional
            Time step for calculations, by default 1.
        penalize_charging : bool, optional
            Flag to penalize charging, by default False.
        """
        super().__init__(
            id, calculate_savings, dt=dt, penalize_charging=penalize_charging
        )

        self.type = "Scipy"
        self.n_v = 0
        self.id_list = set()
        self.batteries: List[Battery] = []

        self.charging_efficiency = 0.95
        self.discharging_efficiency = 0.95
        self.prices = None

        self.cost: Dict = {}

        self.warm_start = False
        self.boundaries = None

        self.energy_arrival = np.array([])
        self.energy_departure = np.array([])
        self.min_energy_contents = np.array([])
        self.max_energy_contents = np.array([])

    def efficiency(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate charging or discharging efficiency based on power direction.

        Parameters
        ----------
        x : np.ndarray
            Input power array.

        Returns
        -------
        np.ndarray
            Efficiency-adjusted power array.
        """
        return np.where(
            x > 0, x * self.charging_efficiency, x / self.discharging_efficiency
        )

    def inverse_efficiency(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate inverse of charging or discharging efficiency based on power direction.

        Parameters
        ----------
        x : np.ndarray
            Input power array.

        Returns
        -------
        np.ndarray
            Inverse efficiency-adjusted power array.
        """
        return np.where(
            x > 0, x / self.charging_efficiency, x * self.discharging_efficiency
        )

    def _initialize_batteries(self) -> None:
        """
        Initialize battery parameters for optimization.
        """
        bnd_battery = []
        energy_max = []
        energy_min = []
        energy_start = []
        energy_end = []

        for bat in self.batteries:
            self.n_v += 1
            bnd_battery.extend(
                [
                    (-bat.power_discharge_max, bat.power_charge_max) if t else (0, 0)
                    for t in bat.connected
                ]
            )
            energy_min.append(bat.energy_min)
            energy_max.append(bat.energy_max)
            energy_start.append(bat.energy_start)
            energy_end.append(bat.energy_end)

        self.boundaries = tuple(bnd_battery)
        self.energy_arrival = np.array(energy_start)
        self.energy_departure = np.array(energy_end)
        self.min_energy_contents = np.array(energy_min)
        self.max_energy_contents = np.array(energy_max)
        self.n_v = len(self.batteries)

    def _cost_spot(self, power: np.ndarray) -> np.ndarray:
        """
        Calculate spot cost based on power consumption.

        Parameters
        ----------
        power : np.ndarray
            Power consumption array.

        Returns
        -------
        np.ndarray
            Spot cost array.
        """
        delta_energy = power * self.dt
        delta_energy = delta_energy.reshape(self.n_v, self.n_t)
        delta_energy_fleet = delta_energy.sum(axis=0)

        sl = 0 if self.site_load is None else self.site_load

        net_export = delta_energy_fleet + sl
        net_export = np.where(net_export < 0, net_export, 0)
        import_reduction = delta_energy_fleet - net_export

        cost_spot = (
            import_reduction * self.tariffs_import + net_export * self.tariffs_export
        )
        return cost_spot * (1 - self.mask_marketed)

    def _cost_marketed(self, power: np.ndarray) -> np.ndarray:
        """
        Calculate penalty for not matching marketed volumes.

        Parameters
        ----------
        power : np.ndarray
            Power consumption array.

        Returns
        -------
        np.ndarray
            Marketed cost penalty array.
        """
        power = power.reshape(self.n_v, self.n_t)
        power_fleet = power.sum(axis=0)
        cost_marketed = np.abs(power_fleet * self.dt - self.marketed_volumes)
        return np.nan_to_num(cost_marketed, 0)

    def _cost_flex(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate flexibility cost.

        Parameters
        ----------
        x : np.ndarray
            Power consumption array.

        Returns
        -------
        np.ndarray
            Flexibility cost array.
        """
        p, n = self._flex(x)
        return -p * self.prices_flex_pos - n * self.prices_flex_neg

    def _cost(self, power: np.ndarray) -> float:
        """
        Calculate total cost.

        Parameters
        ----------
        power : np.ndarray
            Power consumption array.

        Returns
        -------
        float
            Total cost.
        """
        cost = self._cost_spot(power).sum()

        if self.marketed_volumes is not None:
            cost += self._cost_marketed(power).sum()

        if self.prices_flex_pos is not None or self.prices_flex_neg is not None:
            cost += self._cost_flex(power).sum()

        return cost

    def _calc_flex_v(
        self,
        power_v: np.ndarray,
        energy_v: np.ndarray,
        min_energy_content: float,
        max_energy_content: float,
        power_discharge_max: float,
        power_charge_max: float,
        verbose: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate positive and negative flexibility for a single battery.

        Parameters
        ----------
        power_v : np.ndarray
            Power consumption array for a single battery.
        energy_v : np.ndarray
            Energy content array for a single battery.
        min_energy_content : float
            Minimum energy content for the battery.
        max_energy_content : float
            Maximum energy content for the battery.
        power_discharge_max : float
            Maximum discharge power for the battery.
        power_charge_max : float
            Maximum charge power for the battery.
        verbose : int, optional
            Verbosity level, by default 0.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Positive and negative flexibility arrays.
        """
        energy_v_prev = np.array(
            [energy_v[0] - self.efficiency(power_v[0] * self.dt)] + list(energy_v[:-1])
        )

        _flex_pos = np.minimum(
            self.flex_buffer
            * self.inverse_efficiency(
                np.minimum(energy_v, energy_v_prev) - min_energy_content
            )
            / self.dt,
            abs(power_discharge_max) + power_v,
        )

        _flex_neg = np.minimum(
            self.flex_buffer
            * self.inverse_efficiency(
                max_energy_content - np.maximum(energy_v, energy_v_prev)
            )
            / self.dt,
            power_charge_max - power_v,
        )

        if verbose > 0:
            if np.any(_flex_neg < 0):
                print("Flex Negative")
                print(_flex_neg)
                print(max_energy_content)
                print(max_energy_content - energy_v)
                print(max_energy_content - energy_v_prev)
                print(abs(power_discharge_max) + power_v)
            if np.any(_flex_pos < 0):
                print("Flex Positive")
                print(_flex_pos)
                print(energy_v)
                print(energy_v_prev)
                print(power_charge_max - power_v)

        return _flex_pos, _flex_neg

    def _flex(
        self, power: np.ndarray, verbose: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate flexibility for the entire fleet.

        Parameters
        ----------
        power : np.ndarray
            Power consumption array for the entire fleet.
        verbose : int, optional
            Verbosity level, by default 0.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Positive and negative flexibility arrays for the fleet.
        """
        delta_energy = self.efficiency(power.reshape(self.n_v, self.n_t) * self.dt)
        soc = delta_energy.cumsum(axis=1).T + self.energy_arrival
        soc = soc.T

        flex_fleet_pos = []
        flex_fleet_neg = []
        for i, battery in enumerate(self.batteries):
            p, n = self._calc_flex_v(
                power[i * self.n_t : (i + 1) * self.n_t],
                soc[i],
                battery.energy_min,
                battery.energy_max,
                battery.power_discharge_max,
                battery.power_charge_max,
                verbose=verbose,
            )

            flex_fleet_pos.extend(p)
            flex_fleet_neg.extend(n)

        flex_fleet_pos = (
            np.array(flex_fleet_pos).reshape(self.n_v, self.n_t).sum(axis=0)
        )
        flex_fleet_neg = (
            np.array(flex_fleet_neg).reshape(self.n_v, self.n_t).sum(axis=0)
        )

        return flex_fleet_pos, flex_fleet_neg

    def _constraint_lower_limit(self, power: np.ndarray) -> np.ndarray:
        """
        Constraint: State of charge (SOC) cannot go below the lower limit.

        Parameters
        ----------
        power : np.ndarray
            Power consumption array.

        Returns
        -------
        np.ndarray
            Array of constraint values.
        """
        delta_energy = self.efficiency(power) * self.dt
        delta_energy = delta_energy.reshape(self.n_v, self.n_t)
        soc = delta_energy.cumsum(axis=1).T + self.energy_arrival
        result = soc - self.min_energy_contents
        return result.T.reshape(self.n_v * self.n_t)

    def _constraint_upper_limit(self, power: np.ndarray) -> np.ndarray:
        """
        Constraint: State of charge (SOC) cannot go above the upper limit.

        Parameters
        ----------
        power : np.ndarray
            Power consumption array.

        Returns
        -------
        np.ndarray
            Array of constraint values.
        """
        delta_energy = self.efficiency(power) * self.dt
        delta_energy = delta_energy.reshape(self.n_v, self.n_t)
        soc = delta_energy.cumsum(axis=1).T + self.energy_arrival
        result = self.max_energy_contents - soc
        return result.T.reshape(self.n_v * self.n_t)

    def _constraint_fully_charged(self, power: np.ndarray) -> np.ndarray:
        """
        Constraint: Batteries need to be charged to a certain level at the end of the charging cycle.

        Parameters
        ----------
        power : np.ndarray
            Power consumption array.

        Returns
        -------
        np.ndarray
            Array of constraint values.
        """
        delta_energy = self.efficiency(power) * self.dt
        delta_energy = delta_energy.reshape(self.n_v, self.n_t)
        soc_final = delta_energy.sum(axis=1) + self.energy_arrival
        return soc_final - self.energy_departure

    def _constraint_restriction_charge(self, power: np.ndarray) -> np.ndarray:
        """
        Constraint: Charging restriction based on site load.

        Parameters
        ----------
        power : np.ndarray
            Power consumption array.

        Returns
        -------
        np.ndarray
            Array of constraint values.
        """
        sl = 0 if self.site_load is None else self.site_load
        delta_energy = power.reshape(self.n_v, self.n_t) * self.dt
        delta_energy_fleet = delta_energy.sum(axis=0)
        return self.site_load_restriction_charge - (delta_energy_fleet + sl)

    def _constraint_restriction_discharge(self, power: np.ndarray) -> np.ndarray:
        """
        Constraint: Discharging restriction based on site load.

        Parameters
        ----------
        power : np.ndarray
            Power consumption array.

        Returns
        -------
        np.ndarray
            Array of constraint values.
        """
        sl = 0 if self.site_load is None else self.site_load
        delta_energy = power.reshape(self.n_v, self.n_t) * self.dt
        delta_energy_fleet = delta_energy.sum(axis=0)
        return delta_energy_fleet + sl - self.site_load_restriction_discharge

    def optimize(self) -> SiteResult:
        """
        Main optimization function using SciPy's minimize.

        Returns
        -------
        SiteResult
            Optimization result containing dataframe, elapsed time, objective value, and success status.
        """
        self._initialize_batteries()

        constraints = [
            {"type": "ineq", "fun": self._constraint_lower_limit},
            {"type": "ineq", "fun": self._constraint_upper_limit},
            {"type": "ineq", "fun": self._constraint_fully_charged},
        ]
        if self.site_load_restriction_charge is not None:
            constraints.append(
                {"type": "ineq", "fun": self._constraint_restriction_charge}
            )
        if self.site_load_restriction_discharge is not None:
            constraints.append(
                {"type": "ineq", "fun": self._constraint_restriction_discharge}
            )

        start_solving = pd.Timestamp.now()

        x0 = (
            self.get_initial_guess(self.n_v * self.n_t, boundaries=self.boundaries)
            if self.warm_start
            else np.zeros(self.n_v * self.n_t)
        )

        sol = minimize(
            self._cost,
            x0,
            method="SLSQP",
            bounds=self.boundaries,
            options={
                "maxiter": 50,
                "disp": False,
            },
            constraints=constraints,
        )

        solve_time = (pd.Timestamp.now() - start_solving).total_seconds()

        result = sol.x.reshape(self.n_v, self.n_t)

        charging_powers = {
            battery.id: power for (battery, power) in zip(self.batteries, result)
        }

        df = self._calculate_result_df(charging_powers)

        if self.prices_flex_pos is not None:
            p, n = self._flex(sol.x)
            df["site"]["FlexPosFull"] = p
            df["site"]["FlexNegFull"] = n

            df["site"]["FlexPos"] = df["site"]["FlexPosFull"] * np.where(
                self.prices_flex_pos > 0, 1, np.nan
            )
            df["site"]["FlexNeg"] = df["site"]["FlexNegFull"] * np.where(
                self.prices_flex_neg > 0, 1, np.nan
            )

        df["site"]["Objective_Spot"] = self._cost_spot(sol.x)

        if self.marketed_volumes is not None:
            df["site"]["Objective_Marketed_Volumes"] = self._cost_marketed(sol.x)

        if self.prices_flex_pos is not None or self.prices_flex_neg is not None:
            df["site"]["Objective_Flex"] = self._cost_flex(sol.x)

        objective_value = self._cost(sol.x)

        return SiteResult(
            df=df,
            time_elapsed=solve_time,
            objective_value=objective_value,
            success=sol.success,
        )


if __name__ == "__main__":
    prices_import = np.sin(np.linspace(0, 2 * np.pi, 30))
    site_load = 10 * (np.cos(np.linspace(0, 2 * np.pi, 30)) + 1)

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

    marketed_volumes = np.array([2] * 5 + [np.nan] * 25)

    fo = FleetOptimizationSciPy(id=1, calculate_savings=True)
    for battery in [battery1, battery2]:
        fo.add_battery(battery)

    fo.add_prices(prices_import, prices_import)
    res1 = fo.optimize()

    fo.add_marketed_volumes(marketed_volumes)
    res2 = fo.optimize()

    fo.add_flex(
        prices_flex_pos=np.array([0] * 10 + [1] * 4 + [0] * 16),
        prices_flex_neg=np.array([0] * 30),
    )
    res3 = fo.optimize()

    fo.add_site_load(site_load)
    fo.add_site_limits(site_load_restriction_discharge=0)
    res4 = fo.optimize()

    print(res4.df)
    fo.plot(res4, "../figures/test_scipy.png")
