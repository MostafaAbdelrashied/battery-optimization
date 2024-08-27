from math import floor
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from battery_management.assets.battery import Battery
from loguru import logger


class SiteResult:
    """
    Extracts and organizes valuable information from the optimizer.

    Parameters
    ----------
    results : Dict[str, Any]
        Dictionary containing optimization results.

    Attributes
    ----------
    id : str
        Site identifier.
    type : str
        Type of optimization.
    objective_value : float
        Value of the objective function.
    time_elapsed : float
        Time taken for optimization.
    success : bool
        Whether the optimization was successful.
    dt : float
        Time step duration.
    n_t : int
        Number of time steps.
    date_range : pd.DatetimeIndex
        Date range of the optimization.
    batteries : List[Battery]
        List of Battery objects.
    battery_results : pd.DataFrame
        DataFrame containing battery results.
    site_results : pd.DataFrame
        DataFrame containing site results.
    grid_results : pd.DataFrame
        DataFrame containing grid results.
    extra_info : Dict[str, Any]
        Additional information from the optimization.
    savings : pd.DataFrame
        DataFrame containing savings information.
    aggregated_results : pd.DataFrame
        DataFrame containing aggregated results.
    """

    def __init__(self, results: Dict[str, Any]):
        self.id = results.get("id")
        self.type = results.get("type")
        self.objective_value = results.get("objective_value")
        self.time_elapsed = results["solve_time"]
        self.success = results["status"]
        self.dt = results["dt"]
        self.n_t = results["n_t"]
        self.date_range = results["date_range"]
        self.batteries = results["batteries"]

        self.battery_results = results["battery_results"]
        self.site_results = results["site_results"]
        self.grid_results = results["grid_results"]

        self.extra_info = results.get("extra_info", {})

        self.savings = pd.DataFrame()
        if (
            results.get("calculate_savings")
            and results.get("tariffs_import") is not None
        ):
            self.savings = self.calculate_savings()

        self.aggregated_results = self.aggregate_results()
        if results.get("save_file") is not None:
            self.save(results.get("save_file"))

    @classmethod
    def create(
        cls,
        results: Dict[str, Any],
        optimizer: str = "or",
        default_strategy: str = "inactive",
    ) -> "SiteResult":
        """
        Create a SiteResult object based on the optimizer type.

        Parameters
        ----------
        results : Dict[str, Any]
            Optimization results.
        optimizer : str, optional
            Type of optimizer, by default "or".
        default_strategy : str, optional
            Default charging strategy, by default "inactive".

        Returns
        -------
        SiteResult
            An instance of SiteResult or its subclass.
        """
        from battery_management.helper.site_result_or import SiteResultOR

        registered_result_handlers = {"or": SiteResultOR}
        assert (
            optimizer in registered_result_handlers
        ), f"Unknown optimizer {optimizer} caught in SiteResult."
        return registered_result_handlers[optimizer](
            results, default_strategy=default_strategy
        )

    def save(self, filename: str) -> None:
        """
        Save aggregated results to a CSV file.

        Parameters
        ----------
        filename : str
            Path to save the CSV file.
        """
        self.aggregated_results.to_csv(filename)

    def aggregate_results(self) -> pd.DataFrame:
        """
        Aggregate results from site, grid, and savings.

        Returns
        -------
        pd.DataFrame
            DataFrame containing aggregated results.
        """
        agg_results = pd.merge(
            self.site_results,
            self.grid_results,
            how="outer",
            left_index=True,
            right_index=True,
            suffixes=["_site", "_grid"],
        )
        agg_results = pd.merge(
            agg_results, self.savings, how="outer", left_index=True, right_index=True
        )
        return agg_results

    def calculate_savings(self) -> pd.DataFrame:
        """
        Calculate savings compared to generic charging schedules.

        Returns
        -------
        pd.DataFrame
            DataFrame containing savings information.
        """
        savings = pd.DataFrame()

        if "site_load_kw" not in self.site_results.columns:
            feed_from_grid = 0
            feed_into_grid = 0
            site_load_cost = 0
        else:
            site_load = self.site_results["site_load_kw"]
            feed_from_grid = np.where(site_load > 0, site_load, 0)
            feed_into_grid = np.where(site_load <= 0, site_load, 0)
            site_load_cost = feed_from_grid * self.site_results["TariffsImport"]
            if "TariffsExport" in self.site_results.columns:
                site_load_cost += feed_into_grid * self.site_results["TariffsExport"]

        total_load = feed_from_grid + feed_into_grid + self.site_results["power_kw"]
        total_feed_from_grid = np.where(total_load > 0, total_load, 0)
        total_feed_into_grid = np.where(total_load <= 0, total_load, 0)

        savings["cost"] = total_feed_from_grid * self.site_results["TariffsImport"]
        if "TariffsExport" in self.site_results.columns:
            savings["cost"] += total_feed_into_grid * self.site_results["TariffsExport"]
        savings["cost"] -= site_load_cost

        for schedule in ["early", "continuous", "late"]:
            if f"power_kw_{schedule}" in self.site_results.columns:
                total_load = (
                    feed_from_grid
                    + feed_into_grid
                    + self.site_results[f"power_kw_{schedule}"]
                )
                total_feed_from_grid = np.where(total_load > 0, total_load, 0)
                total_feed_into_grid = np.where(total_load <= 0, total_load, 0)

                savings[f"cost_{schedule}"] = (
                    total_feed_from_grid * self.site_results["TariffsImport"]
                )
                if "TariffsExport" in self.site_results.columns:
                    savings[f"cost_{schedule}"] += (
                        total_feed_into_grid * self.site_results["TariffsExport"]
                    )
                savings[f"cost_{schedule}"] -= site_load_cost

                if (
                    "TriadImport" in self.site_results.columns
                    or "TriadExport" in self.site_results.columns
                ):
                    cost_triad = f"cost_triad_{schedule}"
                    savings[cost_triad] = 0
                    if "TriadExport" in self.site_results.columns:
                        savings[cost_triad] += (
                            total_feed_into_grid - feed_into_grid
                        ) * self.site_results["TriadExport"]
                    if "TriadImport" in self.site_results.columns:
                        savings[cost_triad] += (
                            total_feed_from_grid - feed_from_grid
                        ) * self.site_results["TriadImport"]

                savings[f"saving_{schedule}"] = (
                    savings[f"cost_{schedule}"] - savings["cost"]
                )

        return savings

    def _non_optimized_charging(
        self, battery: Battery, late_charging: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Calculate non-optimized charging schedules for comparison.

        Parameters
        ----------
        battery : Battery
            Battery object.
        late_charging : int, optional
            Time step to start late charging, by default None.

        Returns
        -------
        pd.DataFrame
            DataFrame containing non-optimized charging schedules.
        """
        df = pd.DataFrame(index=self.date_range)
        methods = ["early", "continuous"]
        if late_charging is not None:
            methods.append("late")

        for method in methods:
            df[f"power_kw_{method}"] = self.default_charging(
                battery,
                n_t=self.n_t,
                dt=self.dt,
                method=method,
                late_charging=late_charging,
            )
            df[f"energy_content_kwh_{method}"] = (
                df[f"power_kw_{method}"].cumsum() / self.dt + battery.energy_start
            )
        return df

    @staticmethod
    def default_charging(
        battery: Battery, n_t: int, dt: float, method: str = "continuous", **kwargs
    ) -> np.ndarray:
        """
        Calculate default charging schedule for a battery.

        Parameters
        ----------
        battery : Battery
            Battery object.
        n_t : int
            Number of time steps.
        dt : float
            Time step duration.
        method : str, optional
            Charging method, by default "continuous".
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        np.ndarray
            Array containing the charging schedule.
        """
        delta_energy_max = battery.power_charge_max * dt
        delta_energy = battery.energy_end - battery.energy_start
        half_hours_needed = delta_energy / delta_energy_max
        whole_half_hours_needed = floor(half_hours_needed)

        if method == "inactive":
            charging_schedule_kw = [0] * n_t
        elif method == "continuous":
            step = min(delta_energy / n_t, delta_energy_max)
            charging_schedule_kw = [step] * n_t
        elif method == "early":
            energy_remainder = delta_energy - whole_half_hours_needed * delta_energy_max
            charging_schedule_kw = [delta_energy_max] * whole_half_hours_needed + [
                energy_remainder
            ]

            if len(charging_schedule_kw) > n_t:
                charging_schedule_kw = [delta_energy_max] * n_t

            charging_schedule_kw += [0] * (n_t - len(charging_schedule_kw))
        elif method == "late":
            if not kwargs.get("late_charging"):
                logger.warning(
                    "Trying to compute default charging schedule for method 'late' without providing parameter 'late_charging' might lead to unwanted behavior."
                )
            energy_remainder = delta_energy - whole_half_hours_needed * delta_energy_max
            charging_schedule_kw = (
                [0] * (kwargs.get("late_charging", 1) - 1)
                + [delta_energy_max] * whole_half_hours_needed
                + [energy_remainder]
            )
            charging_schedule_kw += [0] * n_t
            charging_schedule_kw = charging_schedule_kw[:n_t]
        else:
            raise ValueError(
                f"Unknown method '{method}' in default_charging. Possible values = inactive, continuous."
            )

        return np.array(charging_schedule_kw) * dt

    def __str__(self) -> str:
        return f"Results:\n - Success: {self.success}\n - Time Elapsed: {self.time_elapsed}\n - Agg. Results: {self.aggregated_results}"
