import os
from abc import ABC
from typing import Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from battery_management.assets.battery import Battery
from battery_management.helper.site_result import SiteResult
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap


class PlottingTool(ABC):
    """
    A tool for creating various plots related to site optimization results.
    """

    @staticmethod
    def _plot_v2g(result: SiteResult, filename: str) -> int:
        """
        Create a set of plots for V2G (Vehicle-to-Grid) results.

        Parameters
        ----------
        result : SiteResult
            The site result object containing optimization results.
        filename : str
            The filename to save the plot.

        Returns
        -------
        int
            Status code (0 for success).
        """
        df = result.aggregated_results
        dt = result.dt

        try:
            df.index = df.index.tz_convert(None)
        except Exception:
            pass

        total_lower_limit = sum(battery.energy_min for battery in result.batteries)
        total_upper_limit = sum(battery.energy_max for battery in result.batteries)

        df["Lower Limit"] = total_lower_limit
        df["Upper Limit"] = total_upper_limit

        if result.extra_info.get("site_load_restriction_charge") is not None:
            df["Restriction Charging"] = result.extra_info[
                "site_load_restriction_charge"
            ]
        if result.extra_info.get("site_load_restriction_discharge") is not None:
            df["Restriction Discharging"] = result.extra_info[
                "site_load_restriction_discharge"
            ]

        cmap = None
        if "TriadImport" in df.columns:
            df["Triad"] = (df["TriadImport"] > 0).astype(int)
            cmap = PlottingTool._create_custom_colormap()

        fig, axes = plt.subplots(3, 4, figsize=(24, 8))

        plot_functions = [
            (PlottingTool._plot_change_in_soc, (axes[0, 0], df, dt, cmap)),
            (PlottingTool._plot_cost_cumulative, (axes[0, 1], df, cmap)),
            (PlottingTool._plot_site_load, (axes[0, 2], df)),
            (PlottingTool._plot_flex_relative, (axes[0, 3], df)),
            (PlottingTool._plot_delta_energy, (axes[1, 0], df, cmap)),
            (PlottingTool._plot_cost, (axes[1, 1], df)),
            (PlottingTool._plot_savings, (axes[1, 2], df)),
            (PlottingTool._plot_flex_absolute, (axes[1, 3], df)),
            (PlottingTool._plot_individual_charging, (axes[2, 0], result, df, cmap)),
            (PlottingTool._plot_energy_price, (axes[2, 1], df)),
            (PlottingTool._plot_triad, (axes[2, 2], df)),
            (PlottingTool._plot_objective, (axes[2, 3], df)),
        ]

        for plot_func, args in plot_functions:
            plot_func(*args)

        fig.subplots_adjust(hspace=0.5)

        title = (
            f"Fleet {result.id}  "
            f"Number of Vehicles: {len(result.batteries)}, "
            f"Optimization Period: {min(result.date_range)} - {max(result.date_range)},  "
            f"N Steps: {result.n_t},  \n"
            f"Optimizer used: {result.type}  "
            f"Total Cost for period: {result.objective_value:.2f}£  "
        )
        fig.suptitle(title)

        plt.savefig(filename)
        plt.close(fig)
        return 0

    @staticmethod
    def _create_custom_colormap() -> LinearSegmentedColormap:
        """
        Create a custom colormap for triad visualization.

        Returns
        -------
        LinearSegmentedColormap
            Custom colormap for triad visualization.
        """
        cmap = plt.cm.jet
        cmaplist = [cmap(i) for i in range(cmap.N)]
        cmaplist[0] = (0.5, 0.5, 0.5, 1.0)
        return LinearSegmentedColormap.from_list("Custom cmap", cmaplist, cmap.N)

    @staticmethod
    def _plot_change_in_soc(
        ax: Axes, df: pd.DataFrame, dt: float, cmap: Optional[LinearSegmentedColormap]
    ) -> None:
        """
        Plot the change in State of Charge (SOC) for batteries.

        Parameters
        ----------
        ax : Axes
            Matplotlib axes object to plot on.
        df : pd.DataFrame
            DataFrame containing the data to plot.
        dt : float
            Time step duration.
        cmap : Optional[LinearSegmentedColormap]
            Custom colormap for triad visualization.
        """
        df_tmp = df.copy()
        energy_initial = (
            df_tmp["energy_content_kwh"][0] - df_tmp["power_kw_site"][0] * dt
        )
        cols = [
            col
            for col in [
                "energy_content_kwh",
                "energy_content_kwh_early",
                "energy_content_kwh_continuous",
                "energy_content_kwh_late",
            ]
            if col in df_tmp
        ]
        idx = df.index[0] - (df.index[1] - df.index[0])
        if isinstance(df_tmp.index, pd.DatetimeIndex):
            idx = pd.Timestamp(idx)
        df_initial = pd.DataFrame({col: [energy_initial] for col in cols}, index=[idx])
        df_tmp = pd.concat([df_initial, df_tmp], axis=0, sort=False)

        ax.set_title("State of Charge [kWh]")
        df_tmp["energy_content_kwh"].plot(label="Optimized", ax=ax)
        for label in ["Early", "Continuous", "Late"]:
            en = f"energy_content_kwh_{label.lower()}"
            if en in df_tmp.columns:
                df_tmp[en].plot(label=label, ax=ax)
        if "Triad" in df_tmp.columns and cmap:
            ax.pcolorfast(
                ax.get_xlim(),
                ax.get_ylim(),
                df_tmp["Triad"].values[np.newaxis],
                cmap=cmap,
                alpha=0.1,
            )
        df["Lower Limit"].plot(alpha=0.5, style="r--", label=None, ax=ax)
        df["Upper Limit"].plot(alpha=0.5, style="r--", label=None, ax=ax)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            [
                handle_val
                for handle_val, legend_val in zip(handles, labels)
                if "Limit" not in legend_val
            ],
            [legend_val for legend_val in labels if "Limit" not in legend_val],
            loc="center left",
            bbox_to_anchor=(-0.5, 0.5),
        )

    @staticmethod
    def _plot_cost_cumulative(
        ax: Axes, df: pd.DataFrame, cmap: Optional[LinearSegmentedColormap]
    ) -> None:
        """
        Plot cumulative cost over time.

        Parameters
        ----------
        ax : Axes
            Matplotlib axes object to plot on.
        df : pd.DataFrame
            DataFrame containing the data to plot.
        cmap : Optional[LinearSegmentedColormap]
            Custom colormap for triad visualization.
        """
        ax.set_title("Cost (Cumulative) [£]")
        cost_cols = [
            col for col in df.columns if ("Cost" in col) and "Triad" not in col
        ]
        if cost_cols:
            for col in cost_cols:
                df[col].cumsum().plot(ax=ax)
            if "Triad" in df.columns and cmap:
                ax.pcolorfast(
                    ax.get_xlim(),
                    ax.get_ylim(),
                    df["Triad"].values[np.newaxis],
                    cmap=cmap,
                    alpha=0.1,
                )
            ax.legend()
        else:
            ax.set_axis_off()

    @staticmethod
    def _plot_site_load(ax: Axes, df: pd.DataFrame) -> None:
        """
        Plot site load over time.

        Parameters
        ----------
        ax : Axes
            Matplotlib axes object to plot on.
        df : pd.DataFrame
            DataFrame containing the data to plot.
        """
        if "Siteload" in df.columns:
            ax.set_title("Site Load [kWh]")
            df["Siteload"].plot(label="Siteload", ax=ax)
            (df["Siteload"] + df["power_kw_site"]).plot(label="Siteload Total", ax=ax)
            if "Restriction Charging" in df.columns:
                df["Restriction Charging"].plot(alpha=0.5, style="r--", ax=ax)
            if "Restriction Discharging" in df.columns:
                df["Restriction Discharging"].plot(alpha=0.5, style="r--", ax=ax)
            ax.legend()
        else:
            ax.set_axis_off()

    @staticmethod
    def _plot_delta_energy(
        ax: Axes, df: pd.DataFrame, cmap: Optional[LinearSegmentedColormap]
    ) -> None:
        """
        Plot change in energy (power) over time.

        Parameters
        ----------
        ax : Axes
            Matplotlib axes object to plot on.
        df : pd.DataFrame
            DataFrame containing the data to plot.
        cmap : Optional[LinearSegmentedColormap]
            Custom colormap for triad visualization.
        """
        ax.set_title("Power [kW]")
        df["power_kw_site"].plot(label="Optimized", ax=ax)
        for label in ["Early", "Continuous", "Late"]:
            if f"power_kw_{label.lower()}" in df.columns:
                df[f"power_kw_{label.lower()}"].plot(label=label, ax=ax)
        if "MarketedVolumes" in df.columns:
            df["MarketedVolumes"].plot(style="kx", label="Marketed", ax=ax)
        if "Triad" in df.columns and cmap:
            ax.pcolorfast(
                ax.get_xlim(),
                ax.get_ylim(),
                df["Triad"].values[np.newaxis],
                cmap=cmap,
                alpha=0.1,
            )
        ax.legend()

    @staticmethod
    def _plot_cost(ax: Axes, df: pd.DataFrame) -> None:
        """
        Plot cost over time.

        Parameters
        ----------
        ax : Axes
            Matplotlib axes object to plot on.
        df : pd.DataFrame
            DataFrame containing the data to plot.
        """
        ax.set_title("Cost [£]")
        cost_cols = [
            col for col in df.columns if ("Cost" in col) and "Triad" not in col
        ]
        if cost_cols:
            for col in cost_cols:
                df[col].plot(ax=ax)
            ax.legend()
        else:
            ax.set_axis_off()

    @staticmethod
    def _plot_savings(ax: Axes, df: pd.DataFrame) -> None:
        """
        Plot savings over time.

        Parameters
        ----------
        ax : Axes
            Matplotlib axes object to plot on.
        df : pd.DataFrame
            DataFrame containing the data to plot.
        """
        ax.set_title("Saving Non-Optimized")
        saving_columns = [
            c
            for c in df.columns
            if c in ["SavingEarly", "SavingContinuous", "SavingLate"]
        ]
        if saving_columns:
            for column in saving_columns:
                df[column].plot(ax=ax)
            ax.legend()
        else:
            ax.set_axis_off()

    @staticmethod
    def _plot_individual_charging(
        ax: Axes,
        result: SiteResult,
        df: pd.DataFrame,
        cmap: Optional[LinearSegmentedColormap],
    ) -> None:
        """
        Plot individual charging schedules for each battery.

        Parameters
        ----------
        ax : Axes
            Matplotlib axes object to plot on.
        result : SiteResult
            SiteResult object containing battery information.
        df : pd.DataFrame
            DataFrame containing the data to plot.
        cmap : Optional[LinearSegmentedColormap]
            Custom colormap for triad visualization.
        """
        df = result.battery_results
        try:
            df.index.tz_convert(None)
        except Exception:
            pass

        ax.set_title("Individual Charging Schedules [kWh]")
        for j, bat in enumerate(result.batteries):
            color = "green" if j == 0 else "red"
            df.loc[bat.id]["power_kw"].plot(
                color=color, alpha=1 / len(result.batteries), ax=ax
            )

        if "Triad" in df.columns and cmap:
            ax.pcolorfast(
                ax.get_xlim(),
                ax.get_ylim(),
                df["Triad"].values[np.newaxis],
                cmap=cmap,
                alpha=0.1,
            )

    @staticmethod
    def _plot_energy_price(ax: Axes, df: pd.DataFrame) -> None:
        """
        Plot energy prices over time.

        Parameters
        ----------
        ax : Axes
            Matplotlib axes object to plot on.
        df : pd.DataFrame
            DataFrame containing the data to plot.
        """
        ax.set_title("Energy Price [£/kWh]")
        if "TariffsImport" in df.columns or "TariffsExport" in df.columns:
            if "TariffsImport" in df.columns:
                ax1 = df["TariffsImport"].plot(label="N2EX", ax=ax)
                ax1.set_ylabel("N2EX")
            if "TriadImport" in df.columns:
                ax2 = df["TriadImport"].plot(label="Triad", secondary_y=True, ax=ax)
                ax2.set_ylabel("Triad")
            ax.legend()
        else:
            ax.set_axis_off()

    @staticmethod
    def _plot_triad(ax: Axes, df: pd.DataFrame) -> None:
        """
        Plot cumulative triad costs over time.

        Parameters
        ----------
        ax : Axes
            Matplotlib axes object to plot on.
        df : pd.DataFrame
            DataFrame containing the data to plot.
        """
        ax.set_title("Triad (Cumulative) [£]")
        if "CostTriadOptimized" in df.columns:
            for col in [
                "CostTriadOptimized",
                "CostTriadEarly",
                "CostTriadContinuous",
                "CostTriadLate",
            ]:
                if col in df.columns:
                    df[col].cumsum().plot(ax=ax)
            ax.legend()
        else:
            ax.set_axis_off()

    @staticmethod
    def _plot_flex_relative(ax: Axes, df: pd.DataFrame) -> None:
        """
        Plot relative flexibility over time.

        Parameters
        ----------
        ax : Axes
            Matplotlib axes object to plot on.
        df : pd.DataFrame
            DataFrame containing the data to plot.
        """
        ax.set_title("Flexibility Relative [kWh]")
        if "FlexPos" in df.columns and "FlexNeg" in df.columns:
            (df["power_kw_site"] - df["FlexPos"]).plot(style="r--", ax=ax)
            (df["power_kw_site"] + df["FlexNeg"]).plot(style="r--", ax=ax)
            df["power_kw_site"].plot(color="red", ax=ax)
            plt.fill_between(
                df.index,
                (df["power_kw_site"] - df["FlexPos"]),
                (df["power_kw_site"] + df["FlexNeg"]),
                facecolor="red",
                alpha=0.2,
                interpolate=True,
            )
            ax.legend()
        else:
            ax.set_axis_off()

    @staticmethod
    def _plot_flex_absolute(ax: Axes, df: pd.DataFrame) -> None:
        """
        Plot absolute flexibility over time.

        Parameters
        ----------
        ax : Axes
            Matplotlib axes object to plot on.
        df : pd.DataFrame
            DataFrame containing the data to plot.
        """
        ax.set_title("Flexibility Absolute [kWh]")
        if "FlexPos" in df.columns and "FlexNeg" in df.columns:
            df["FlexPosFull"].plot(style="r--", ax=ax)
            df["FlexPos"].plot(style="r-", ax=ax)
            df["FlexNegFull"].plot(style="b--", ax=ax)
            df["FlexNeg"].plot(style="b-", ax=ax)
            ax.legend()
        else:
            ax.set_axis_off()

    @staticmethod
    def _plot_objective(ax: Axes, df: pd.DataFrame) -> None:
        """
        Plot objective function values over time.

        Parameters
        ----------
        ax : Axes
            Matplotlib axes object to plot on.
        df : pd.DataFrame
            DataFrame containing the data to plot.
        """
        ax.set_title("Objective [£]")
        objective_cols = [col for col in df.columns if "Objective" in col]
        if objective_cols:
            for col in objective_cols:
                df[col].plot(ax=ax)
            ax.legend()
        else:
            ax.set_axis_off()

    @staticmethod
    def _plot_debug(result: SiteResult, filename: str) -> None:
        """
        Create a debug plot for the optimization result.

        Parameters
        ----------
        result : SiteResult
            SiteResult object containing optimization results.
        filename : str
            The filename to save the plot.
        """
        suptitle = f"{result.batteries} CPs –"
        df = result.aggregated_results
        bat_df = result.battery_results

        fig, axes = plt.subplots(5, 1, figsize=(15, 15))

        PlottingTool._plot_debug_connected(axes[0], result.batteries)
        PlottingTool._plot_debug_power(axes[1], bat_df, result.batteries)
        PlottingTool._plot_debug_total_load(axes[2], df, result)
        PlottingTool._plot_debug_energy(axes[3], bat_df, result.batteries)
        PlottingTool._plot_debug_price(axes[4], df)

        plt.suptitle(suptitle)
        plt.savefig(f"{filename.split('.png')[0]}_debugging.png")
        plt.close(fig)

    @staticmethod
    def _plot_debug_connected(ax: Axes, batteries: List[Battery]) -> None:
        """Plot connectivity of batteries."""
        for bat in batteries:
            ax.plot(bat.connected, label=bat.id)
        ax.set_title("Connected")
        ax.legend()

    @staticmethod
    def _plot_debug_power(
        ax: Axes, bat_df: pd.DataFrame, batteries: List[Battery]
    ) -> None:
        """Plot power for each battery."""
        for bat in batteries:
            ax.plot(bat_df.loc[bat.id]["power_kw"], label=bat.id)
        ax.set_title("Power")
        ax.legend()

    @staticmethod
    def _plot_debug_total_load(ax: Axes, df: pd.DataFrame, result: SiteResult) -> None:
        """
        Plot total load for debugging purposes.

        Parameters
        ----------
        ax : Axes
            Matplotlib axes object to plot on.
        df : pd.DataFrame
            DataFrame containing the data to plot.
        result : SiteResult
            SiteResult object containing optimization results.
        """
        ax.plot(df["power_kw_site"], label="CP")
        if "site_load_kw" in df.columns:
            ax.plot(df["site_load_kw"], label="site")
            ax.plot((df["site_load_kw"] + df["power_kw_site"]), label="total")
        if result.extra_info.get("site_load_restriction_charge") is not None:
            site_load_restriction_charge = result.extra_info[
                "site_load_restriction_charge"
            ]
            ax.axhline(
                site_load_restriction_charge, c="k", ls=":", label="limit_purchase"
            )
            if result.extra_info.get("site_constraint_purchase") is not None:
                site_constraint_purchase = result.extra_info["site_constraint_purchase"]
                ax.axhline(
                    site_constraint_purchase, c="b", ls=":", label="new_limit_purchase"
                )
        ax.set_title("Total Load")
        ax.legend()

    @staticmethod
    def _plot_debug_energy(
        ax: Axes, bat_df: pd.DataFrame, batteries: List[Battery]
    ) -> None:
        """
        Plot energy content for debugging purposes.

        Parameters
        ----------
        ax : Axes
            Matplotlib axes object to plot on.
        bat_df : pd.DataFrame
            DataFrame containing battery data.
        batteries : List[Battery]
            List of Battery objects.
        """
        for bat in batteries:
            ax.plot(bat_df.loc[bat.id]["energy_content_kwh"], label=bat.id)
        ax.set_title("Energy")
        ax.legend()

    @staticmethod
    def _plot_debug_price(ax: Axes, df: pd.DataFrame) -> None:
        """
        Plot energy prices for debugging purposes.

        Parameters
        ----------
        ax : Axes
            Matplotlib axes object to plot on.
        df : pd.DataFrame
            DataFrame containing the data to plot.
        """
        ax.plot(df["TariffsImport"], label="Volume")
        if "capacity_tariffs_import" in df.columns:
            ax.axhline(df["capacity_tariffs_import"], label="Capacity")
        ax.set_title("Price")
        ax.legend()

    @staticmethod
    def _plot_user_friendly(result: SiteResult, filename: str) -> int:
        """
        Create a user-friendly plot for the optimization result.

        Parameters
        ----------
        result : SiteResult
            SiteResult object containing optimization results.
        filename : str
            The filename to save the plot.

        Returns
        -------
        int
            Status code (0 for success, 1 for failure).
        """
        if "site_load_kw" not in result.aggregated_results:
            return 1

        suptitle = f"{result.batteries} CPs –"
        df = result.aggregated_results
        site_load_index = df.index
        site_load = df["site_load_kw"]
        site_load_restriction_charge = result.extra_info.get(
            "site_load_restriction_charge"
        )
        site_constraint_purchase = result.extra_info.get("site_constraint_purchase")

        fig, ax = plt.subplots(figsize=(15, 10))
        ax.set_facecolor("#0000")
        plt.grid(False)

        plt.fill_between(site_load_index, 0, site_load, color="b", alpha=0.8)
        ymax = site_load.max()

        if site_load_restriction_charge is not None:
            ymax = site_load_restriction_charge + 1
            plt.axhline(
                site_load_restriction_charge, c="#b00402", label="limit_purchase"
            )
            plt.fill_between(
                site_load_index,
                site_load.max(),
                site_load_restriction_charge,
                color=(205 / 256, 205 / 256, 205 / 256),
                alpha=0.3,
            )
            if site_constraint_purchase is not None:
                plt.axhline(site_constraint_purchase, c="k", label="new_limit_purchase")
                plt.fill_between(
                    site_load_index,
                    site_load_restriction_charge,
                    site_constraint_purchase,
                    color="#b00402",
                    alpha=0.3,
                )
                if site_constraint_purchase > site_load_restriction_charge:
                    ymax = site_constraint_purchase
                    suptitle += f" Grid Connection NOT respected: {site_constraint_purchase:.2f} kW instead of {site_load_restriction_charge} kW"
                else:
                    suptitle += f" Grid Connection respected: {site_constraint_purchase:.2f} kW = {site_load_restriction_charge} kW"

        plt.fill_between(
            site_load_index, 0, site_load.max(), color="#1ea2b1", alpha=0.6
        )
        plt.axhline(site_load.max(), c="#1ea2b1", label="peak", lw=2)
        plt.legend()
        plt.xlim(site_load_index.min(), site_load_index.max())
        plt.ylim(0, ymax)
        plt.title(suptitle)
        plt.savefig(f"{filename.split('.png')[0]}_nice.png")
        plt.close(fig)
        return 0

    @staticmethod
    def plot_dispatcher(
        day_ahead_sites: List[Any],
        datetime_index: pd.DatetimeIndex,
        ts: pd.Timestamp,
        achieved_volumes: np.ndarray,
        x: Optional[np.ndarray],
        pool_marketed_volumes: np.ndarray,
        title_info: str = "",
        filename_info: str = "",
    ) -> None:
        """
        Create dispatcher plots for optimization results.

        Parameters
        ----------
        day_ahead_sites : List[Any]
            List of day-ahead site objects.
        datetime_index : pd.DatetimeIndex
            DatetimeIndex for the x-axis.
        ts : pd.Timestamp
            Timestamp for the plot.
        achieved_volumes : np.ndarray
            Array of achieved volumes.
        x : Optional[np.ndarray]
            Array of dispatched volumes (optional).
        pool_marketed_volumes : np.ndarray
            Array of pool marketed volumes.
        title_info : str, optional
            Additional information for the plot title.
        filename_info : str, optional
            Additional information for the filename.
        """
        _path = f"reports/figures/global_dispatcher/{ts.date()}/"
        os.makedirs(_path, exist_ok=True)

        if x is not None:
            fig, ax = plt.subplots(figsize=(20, 5))
            for i, site in enumerate(day_ahead_sites):
                data = (
                    pd.DataFrame(
                        {"Dispatched": x[i, :], "Achieved": achieved_volumes[i, :]},
                        index=datetime_index,
                    )
                    .resample("1min")
                    .pad()
                )
                ax.plot(
                    data.index,
                    data["Dispatched"],
                    "ro-",
                    label=f"Dispatched {site.site.site_id}",
                )
                ax.plot(
                    data.index,
                    data["Achieved"],
                    "bx-",
                    label=f"Achieved {site.site.site_id}",
                )
            ax.set_title(f"Achieved volumes {title_info}")
            ax.legend()
            plt.savefig(f"{_path}dispatch_achieved_volumes_{filename_info}.png")
            plt.close(fig)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))

        if x is not None:
            volumes_summed = np.zeros(x.shape[1])
            for i, site in enumerate(day_ahead_sites):
                before = volumes_summed.copy()
                volumes_summed += x[i, :]
                ax1.fill_between(
                    datetime_index,
                    before,
                    volumes_summed,
                    label=f"Dispatched {site.site.site_id}",
                )
            ax1.plot(
                datetime_index, pool_marketed_volumes, "k--", label="Marketed Pool"
            )
            ax1.set_title(f"Dispatched volumes {title_info}")
            ax1.legend()

        volumes_summed = (
            pd.DataFrame(
                {"Achieved": np.zeros(achieved_volumes.shape[1])}, index=datetime_index
            )
            .resample("1min")
            .pad()
        )
        for i, site in enumerate(day_ahead_sites):
            before = volumes_summed.copy()
            data = (
                pd.DataFrame({"Achieved": achieved_volumes[i, :]}, index=datetime_index)
                .resample("1min")
                .pad()
            )
            volumes_summed["Achieved"] += data["Achieved"]
            ax2.fill_between(
                data.index,
                before["Achieved"],
                volumes_summed["Achieved"],
                label=f"Achieved {site.site.site_id}",
            )

        pool_marketed_volumes = (
            pd.DataFrame({"Marketed": pool_marketed_volumes}, index=datetime_index)
            .resample("1min")
            .pad()
        )
        ax2.plot(
            pool_marketed_volumes.index,
            pool_marketed_volumes["Marketed"],
            "k--",
            label="Marketed Pool",
        )
        ax2.set_title(f"Achieved volumes {title_info}")
        ax2.legend()

        plt.savefig(f"{_path}/sum_dispatched_{filename_info}.png")
        plt.close(fig)
