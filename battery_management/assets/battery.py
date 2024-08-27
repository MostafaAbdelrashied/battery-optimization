from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class Battery:
    """
    Represents a battery with various parameters for optimization.
    """

    id: int
    capacity: float
    energy_min: float
    energy_max: float
    power_charge_max: float
    power_discharge_max: float
    connected: List[bool]
    energy_start: float = field(default=None)
    energy_end: float = field(default=None)
    power_charge_min: float = 0
    efficiency_charge: float = 0.95
    efficiency_discharge: float = 0.95
    stationary: bool = False
    cycle_life: int = 5000
    cycle_usage: int = 0
    battery_costs: float = 0
    affected_charging_point_id: Optional[int] = None

    def __post_init__(self):
        if self.energy_start is None:
            self.energy_start = self.energy_min
        if self.energy_end is None:
            self.energy_end = self.energy_min

        self._validate_energy(self.energy_start, "Energy at start")
        self._validate_energy(self.energy_end, "Energy at end")

        if self.power_discharge_max != 0 and self.power_charge_min != 0:
            raise ValueError(
                "Cannot set a value to power_charge_min if power_discharge_max is not 0."
            )

        self._calculate_cycle_costs()

    def _validate_energy(self, energy: float, description: str) -> None:
        """
        Validate that the given energy is within the battery limits.
        """
        if not self.energy_min <= energy <= self.energy_max:
            raise ValueError(
                f"{description} ({energy}) outside of battery limits [{self.energy_min}, {self.energy_max}]"
            )

    def _calculate_cycle_costs(self) -> None:
        """
        Calculate cycle costs for the battery.
        """
        self.cycle_cost = self.battery_costs / self.cycle_life
        self.cycle_cost_per_kwh = self.cycle_cost / (2 * self.capacity)

    def info(self) -> str:
        """
        Get a string representation of the battery's information.
        """
        return (
            f"{'-' * 25}\n"
            f"Battery {self.id} \n"
            f"Allowed Energy [{self.energy_min}-{self.energy_max}] \n"
            f"Energy Beginning/End [{self.energy_start}-{self.energy_end}] \n"
        )

    def __repr__(self) -> str:
        return f"Battery(id={self.id}, capacity={self.capacity})"

    def is_connected(self) -> bool:
        """
        Check if the battery is connected at any point.
        """
        return any(self.connected)

    def has_single_charging_session(self) -> bool:
        df = pd.DataFrame({"connected": np.int64(self.connected)})
        diffs = df["connected"].diff()
        start_count = sum(diffs == 1)
        end_count = sum(diffs == -1)

        if start_count > 1 or end_count > 1:
            return False

        if start_count == 1 and end_count == 1:
            start_idx = np.where(diffs == 1)[0][0]
            end_idx = np.where(diffs == -1)[0][0]
            return start_idx < end_idx

        return True

    def add_cycle_costs(self, battery_costs: float, cycle_life: int = 5000) -> None:
        """
        Add cycle costs to the battery.
        """
        self.cycle_life = cycle_life
        self.battery_costs = battery_costs
        self._calculate_cycle_costs()


if __name__ == "__main__":
    battery = Battery(
        id=42,
        energy_start=12,
        energy_end=40,
        capacity=40,
        energy_min=5,
        energy_max=40,
        power_charge_max=5,
        power_discharge_max=5,
        connected=[False] * 5 + [True] * 25,
    )

    print(battery.info())
    print(battery)
