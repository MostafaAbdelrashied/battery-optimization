from typing import Any, Dict

import numpy as np
import pandas as pd
from battery_management.helper.site_result import SiteResult


class SiteResultOR(SiteResult):
    """
    Extracts valuable information from the OR optimizer to be consumed by the standard SiteResult class.

    Parameters
    ----------
    results : Dict[str, Any]
        Dictionary containing optimization results.
    default_strategy : str, optional
        Default charging strategy, by default "inactive".

    Attributes
    ----------
    batteries : List[Battery]
        List of Battery objects.
    n_t : int
        Number of time steps.
    dt : float
        Time step duration.
    date_range : pd.DatetimeIndex
        Date range of the optimization.
    battery_results : pd.DataFrame
        DataFrame containing battery results.
    site_results : pd.DataFrame
        DataFrame containing site results.
    grid_results : pd.DataFrame
        DataFrame containing grid results.
    extra_info : Dict[str, Any]
        Additional information from the optimization.
    """

    def __init__(self, results: Dict[str, Any], default_strategy: str = "inactive"):
        self.batteries = results["batteries"]
        self.n_t = results["n_t"]
        self.dt = results["dt"]
        self.date_range = results["date_range"]

        self.battery_results = self.calculate_battery_results(results, default_strategy)
        self.site_results = self.calculate_site_results(results)
        self.grid_results = self.calculate_grid_results(results)

        self.extra_info = self.get_extra_info(results)
        results.update(self.__dict__)
        super().__init__(results)

    def calculate_battery_results(
        self, results: Dict[str, Any], default_strategy: str
    ) -> pd.DataFrame:
        """
        Calculate battery results from optimization results.

        Parameters
        ----------
        results : Dict[str, Any]
            Optimization results.
        default_strategy : str
            Default charging strategy.

        Returns
        -------
        pd.DataFrame
            DataFrame containing battery results.
        """
        battery_results = pd.DataFrame()
        for battery in results["batteries"]:
            bat_id = battery.id
            _results = pd.DataFrame(index=self.date_range)

            if results["status"] == 0:
                _results["power_kw"] = [
                    results["fleet_power"][battery.id]["charge"][t].SolutionValue()
                    - results["fleet_power"][battery.id]["discharge"][t].SolutionValue()
                    for t in range(self.n_t)
                ]
            else:
                _results["power_kw"] = self.default_charging(
                    battery, method=default_strategy, n_t=self.n_t, dt=self.dt
                )

            _results["energy_content_kwh"] = (
                _results["power_kw"].cumsum() * self.dt + battery.energy_start
            )
            _results["soc_perc"] = _results["energy_content_kwh"] / battery.capacity
            _results["battery_id"] = bat_id

            if results.get("calculate_savings_non_optimized"):
                _results = pd.concat(
                    [_results, self._non_optimized_charging(battery)], axis=1
                )

            if results.get("flex_pos"):
                _results["flex_pos"] = [
                    results["flex_pos"][bat_id][t].SolutionValue()
                    for t in range(self.n_t)
                ]
            if results.get("flex_neg"):
                _results["flex_neg"] = [
                    results["flex_neg"][bat_id][t].SolutionValue()
                    for t in range(self.n_t)
                ]

            battery_results = battery_results.append(_results, ignore_index=False)

        battery_results = (
            battery_results.reset_index()
            .rename(columns={"index": "time"})
            .set_index(["battery_id", "time"], verify_integrity=True)
        )
        return battery_results

    def calculate_site_results(self, results: Dict[str, Any]) -> pd.DataFrame:
        """
        Calculate site results from optimization results.

        Parameters
        ----------
        results : Dict[str, Any]
            Optimization results.

        Returns
        -------
        pd.DataFrame
            DataFrame containing site results.
        """
        site_results = pd.DataFrame(index=self.date_range)
        site_results.index.name = "time"
        site_power_kw_optimizer = [
            results["sum_power_fleet"]["charge"][t].solution_value()
            - results["sum_power_fleet"]["discharge"][t].solution_value()
            for t in range(self.n_t)
        ]

        site_results["power_kw"] = (
            self.battery_results["power_kw"].groupby(["time"]).sum()
        )
        site_results["energy_content_kwh"] = (
            self.battery_results["energy_content_kwh"].groupby(["time"]).sum()
        )

        if results.get("calculate_savings_non_optimized"):
            methods = [
                col.split("_")[2]
                for col in self.battery_results.columns
                if "power_kw" in col and len(col.split("_")) == 3
            ]
            for m in methods:
                site_results[f"power_kw_{m}"] = (
                    self.battery_results[f"power_kw_{m}"].groupby(["time"]).sum()
                )
                site_results[f"energy_content_kwh_{m}"] = (
                    self.battery_results[f"energy_content_kwh_{m}"]
                    .groupby(["time"])
                    .sum()
                )

        assert np.isclose(
            np.sum(np.abs(site_power_kw_optimizer)),
            np.sum(np.abs(site_results["power_kw"])),
        )

        for source_name, target_name in {
            "tariffs_import": "TariffsImport",
            "tariffs_export": "TariffsExport",
            "triad_tariffs_import": "TriadImport",
            "triad_tariffs_export": "TriadExport",
            "site_load": "site_load_kw",
            "prices_flex_neg": "PricesFlexNeg",
            "prices_flex_pos": "PricesFlexPos",
            "marketed_flex_neg": "MarketedFlexNeg",
            "marketed_flex_pos": "MarketedFlexPos",
            "marketed_volumes": "MarketedVolumes",
        }.items():
            if results[source_name] is not None:
                site_results[target_name] = results[source_name]

        if "site_load_kw" in site_results.columns:
            site_results["total_load_kw"] = (
                site_results["site_load_kw"] + site_results["power_kw"]
            )

        opt_costs = results["cost"]
        cost2var_names = {
            "FlexPos": "flex_pos_total",
            "FlexNeg": "flex_neg_total",
        }
        for cost_name, var_name in cost2var_names.items():
            if cost_name in opt_costs:
                site_results[cost_name] = [
                    results[var_name][t].SolutionValue() for t in range(self.n_t)
                ]

        for source_name, target_name in {
            "Triad": "CostTriadOptimized",
            "penalty_charging": "penalty_charging",
            "Spot": "SpotCost",
        }.items():
            if source_name in opt_costs:
                site_results[target_name] = [
                    opt_costs[source_name][t].SolutionValue() for t in range(self.n_t)
                ]

        return site_results

    def calculate_grid_results(self, results: Dict[str, Any]) -> pd.DataFrame:
        """
        Calculate grid results from optimization results.

        Parameters
        ----------
        results : Dict[str, Any]
            Optimization results.

        Returns
        -------
        pd.DataFrame
            DataFrame containing grid results.
        """
        grid_results = pd.DataFrame(index=self.date_range)
        grid_results["power_kw"] = [
            results["grid_power"]["purchase"][t].SolutionValue()
            - results["grid_power"]["feed"][t].SolutionValue()
            for t in range(self.n_t)
        ]
        grid_results["curtailed_power_kw"] = [
            results["grid_power"]["curtail"][t].SolutionValue() for t in range(self.n_t)
        ]
        return grid_results

    @staticmethod
    def get_extra_info(results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract additional information from optimization results.

        Parameters
        ----------
        results : Dict[str, Any]
            Optimization results.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing extra information.
        """
        extra_info = {
            "debug": {
                "disconnected": {
                    bat.id: [
                        results["fleet_bool"][bat.id]["disconnected"][t].SolutionValue()
                        for t in range(results["n_t"])
                    ]
                    for bat in results["batteries"]
                }
            }
        }

        if results.get("site_constraint", {}).get("purchase") is not None:
            extra_info["site_constraint_purchase"] = results["site_constraint"][
                "purchase"
            ].SolutionValue()

        return extra_info

    def dict(self) -> Dict[str, Any]:
        """
        Convert the object to a dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation of the object.
        """
        return self.__dict__
