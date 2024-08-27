from typing import Any, Dict

import pandas as pd
from loguru import logger


class PoolResult:
    """
    Represents the results of a pool optimization.

    Parameters
    ----------
    results_dict : Dict[str, Any]
        Dictionary containing site results.

    Attributes
    ----------
    site_result_dictionary : Dict[str, Any]
        Stores the full information from the site results.
    site_results : pd.DataFrame
        DataFrame containing aggregated site results.
    battery_results : pd.DataFrame
        DataFrame containing aggregated battery results.
    pool_results : pd.DataFrame
        DataFrame containing aggregated pool results.
    """

    def __init__(self, results_dict: Dict[str, Any]):
        logger.debug("Transform BOP result into VPP format")
        self.site_result_dictionary = results_dict
        self.site_results = pd.DataFrame()
        self.battery_results = pd.DataFrame()
        self.pool_results = pd.DataFrame()

        for site_id, site_result in results_dict.items():
            df_site = site_result.site_results.copy()
            df_site["site_id"] = site_id
            self.site_results = pd.concat([self.site_results, df_site])

            next_battery = site_result.battery_results.copy()
            next_battery["site_id"] = site_id
            self.battery_results = pd.concat([self.battery_results, next_battery])

        self.pool_results = self.site_results.groupby("time").sum()
        self.pool_results = self.pool_results.drop(columns="site_id", errors="ignore")

    @staticmethod
    def concatenate_results_site(
        result1: Dict[str, Any] = None, result2: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Concatenate two result dictionaries for the same time period but different sites.

        Parameters
        ----------
        result1 : Dict[str, Any], optional
            First result dictionary.
        result2 : Dict[str, Any], optional
            Second result dictionary.

        Returns
        -------
        Dict[str, Any]
            Concatenated result dictionary.
        """
        if result2 is None:
            return result1
        if result1 is None:
            return result2

        result = {
            "batteries": {**result1["batteries"], **result2["batteries"]},
            "sites": {**result1["sites"], **result2["sites"]},
            "pool": result1["pool"].add(result2["pool"], fill_value=0),
        }

        return result

    @staticmethod
    def concatenate_results_time(
        result1: Dict[str, Any] = None, result2: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Concatenate two result dictionaries for different time periods.

        Parameters
        ----------
        result1 : Dict[str, Any], optional
            First result dictionary.
        result2 : Dict[str, Any], optional
            Second result dictionary.

        Returns
        -------
        Dict[str, Any]
            Concatenated result dictionary.
        """
        if result2 is None:
            return result1
        if result1 is None:
            return result2

        result = {"sites": {}, "batteries": {}}

        for i in result1["batteries"].keys():
            df = pd.concat(
                [result1["batteries"][i], result2["batteries"][i]], sort=False
            )
            df = df.loc[~df.index.duplicated(keep="first")]
            result["batteries"][i] = df.sort_index()

        for i in result1["sites"].keys():
            df = pd.concat([result1["sites"][i], result2["sites"][i]], sort=False)
            df = df.loc[~df.index.duplicated(keep="first")]
            result["sites"][i] = df.sort_index()

        df = pd.concat([result1["pool"], result2["pool"]], sort=False)
        df = df.loc[~df.index.duplicated(keep="first")]
        result["pool"] = df.sort_index()

        return result

    def __str__(self) -> str:
        return f"PoolResult: {self.pool_results.head()}"
