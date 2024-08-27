from typing import Dict, List, Optional

import pandas as pd
from battery_management.assets.charging_point import ChargingPoint
from battery_management.assets.grid import Grid
from battery_management.assets.stationary_battery import StationaryBattery


class Site:
    """
    Represents a site in the VPP (Virtual Power Plant) definition.

    Parameters
    ----------
    site : Dict
        A dictionary containing site information and components.

    Attributes
    ----------
    site_id : int
        Unique identifier for the site.
    n_charging_points : Optional[int]
        Number of charging points at the site.
    country : Optional[str]
        Country where the site is located.
    voltage_level : Optional[float]
        Voltage level of the site.
    charging_points : List[ChargingPoint]
        List of charging points at the site.
    stationary_batteries : List[StationaryBattery]
        List of stationary batteries at the site.
    grid : Optional[Grid]
        Grid information for the site.
    siteload_restriction_half_hour_charge : Optional[float]
        Half-hour charge restriction for the site load.
    siteload_restriction_half_hour_discharge : Optional[float]
        Half-hour discharge restriction for the site load.
    site_load_components : List[str]
        List of components contributing to the site load.
    site_load : pd.DataFrame
        DataFrame containing site load information.
    """

    def __init__(self, site: Dict):
        self.site_id: int = site["site_id"]
        self.n_charging_points: Optional[int] = site.get("n_charging_points")
        self.country: Optional[str] = site.get("country")
        self.voltage_level: Optional[float] = site.get("voltage_level")

        self.charging_points: List[ChargingPoint] = [
            self._create_charging_point(c) for c in site.get("charging_points", [])
        ]
        self.stationary_batteries: List[StationaryBattery] = [
            self._create_stationary_battery(b)
            for b in site.get("stationary_batteries", [])
        ]

        self.grid: Optional[Grid] = Grid(**site["grid"]) if site.get("grid") else None

        self.siteload_restriction_half_hour_charge: Optional[float] = site.get(
            "siteload_restriction_half_hour_charge"
        )
        self.siteload_restriction_half_hour_discharge: Optional[float] = site.get(
            "siteload_restriction_half_hour_discharge"
        )

        self.site_load_components: List[str] = site.get("site_load_components", [])
        self.site_load: pd.DataFrame = pd.DataFrame()

    @staticmethod
    def _create_charging_point(cp: Dict) -> ChargingPoint:
        """
        Create a ChargingPoint object from a dictionary.

        Parameters
        ----------
        cp : Dict
            Dictionary containing charging point information.

        Returns
        -------
        ChargingPoint
            A ChargingPoint object.
        """
        return ChargingPoint(cp)

    @staticmethod
    def _create_stationary_battery(battery: Dict) -> StationaryBattery:
        """
        Create a StationaryBattery object from a dictionary.

        Parameters
        ----------
        battery : Dict
            Dictionary containing battery information.

        Returns
        -------
        StationaryBattery
            A StationaryBattery object.
        """
        return StationaryBattery(**battery)

    def __repr__(self) -> str:
        return f"Site {self.site_id}"

    def __str__(self) -> str:
        return (
            f"Site {self.site_id}\n"
            f"- Stationary Batteries: {self.stationary_batteries}\n"
            f"- Charging Points: {self.charging_points}"
        )


if __name__ == "__main__":
    site1 = {"site_id": 1}
    site = Site(site1)
    print(site)

    stat_bat1 = dict(
        id=42, energy_min=5, energy_max=40, power_charge_max=5, power_discharge_max=5
    )
    site1["stationary_batteries"] = [stat_bat1]
    site = Site(site1)
    print(site)

    cp1 = {
        "asset_id": 1741,
        "charging_power_kw": 65,
        "discharging_power_kw": 60,
        "charging_efficiency": 0.95,
        "discharging_efficiency": 0.95,
        "ffr_pos_pq_kw": 0,
        "ffr_neg_pq_kw": 0,
        "dno_zone": 12,
        "allocated": [],
        "dayahead_tariff": [],
        "third_party_costs": [],
    }
    site1["charging_points"] = [cp1]
    site = Site(site1)
    print(site)

    print([site])
