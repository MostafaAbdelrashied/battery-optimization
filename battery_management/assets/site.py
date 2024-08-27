from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from battery_management.assets.charging_point import ChargingPoint
from battery_management.assets.grid import Grid
from battery_management.assets.stationary_battery import StationaryBattery


@dataclass
class Site:
    """
    Represents a site in the VPP (Virtual Power Plant) definition.

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

    site_id: int
    n_charging_points: Optional[int] = None
    country: Optional[str] = None
    voltage_level: Optional[float] = None
    charging_points: List[ChargingPoint] = field(default_factory=list)
    stationary_batteries: List[StationaryBattery] = field(default_factory=list)
    grid: Optional[Grid] = None
    siteload_restriction_half_hour_charge: Optional[float] = None
    siteload_restriction_half_hour_discharge: Optional[float] = None
    site_load_components: List[str] = field(default_factory=list)
    site_load: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self):
        if self.site_id < 0:
            raise ValueError("site_id must be non-negative")
        if self.n_charging_points is not None and self.n_charging_points < 0:
            raise ValueError("n_charging_points must be non-negative")

    def add_charging_point(self, charging_point: ChargingPoint) -> None:
        """
        Add a charging point to the site.

        Parameters
        ----------
        charging_point : ChargingPoint
            The charging point to add.
        """
        self.charging_points.append(charging_point)
        if self.n_charging_points is not None:
            self.n_charging_points += 1

    def add_stationary_battery(self, battery: StationaryBattery) -> None:
        """
        Add a stationary battery to the site.

        Parameters
        ----------
        battery : StationaryBattery
            The stationary battery to add.
        """
        self.stationary_batteries.append(battery)

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
    site = Site(**site1)
    print(site)

    stat_bat1 = dict(
        id=42,
        energy_min=5,
        energy_max=40,
        power_charge_max=5,
        power_discharge_max=5,
        capacity=40,
    )
    site1["stationary_batteries"] = [stat_bat1]
    site = Site(**site1)
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
    site = Site(**site1)
    print(site)

    print([site])
