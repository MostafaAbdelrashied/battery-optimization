from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd


@dataclass
class Battery:
    id: int
    energy_start: float
    energy_end: float
    energy_min: float
    energy_max: float
    power_charge_max: float
    power_discharge_max: float
    connected: List[bool]
    capacity: float
    power_charge_min: float = 0
    efficiency_charge: float = 1.0
    efficiency_discharge: float = 1.0
    stationary: bool = False
    cycle_life: int = 5000
    cycle_usage: int = 0
    battery_costs: float = 0
    affected_charging_point_id: int = None
    cycle_cost: float = field(init=False)
    cycle_cost_per_kwh: float = field(init=False)

    def __post_init__(self):
        self._validate_energy(self.energy_start, "Energy at start")
        self._validate_energy(self.energy_end, "Energy at end")
        self._validate_power_settings()
        self._calculate_cycle_costs()

    def _validate_energy(self, energy: float, description: str) -> None:
        if not self.energy_min <= energy <= self.energy_max:
            raise ValueError(
                f"{description} ({energy}) outside of battery limits [{self.energy_min}, {self.energy_max}]"
            )

    def _validate_power_settings(self):
        if self.power_discharge_max != 0 and self.power_charge_min != 0:
            raise ValueError(
                "Cannot set a value to power_charge_min if power_discharge_max is not 0."
            )

    def _calculate_cycle_costs(self):
        self.cycle_cost = self.battery_costs / self.cycle_life
        self.cycle_cost_per_kwh = self.cycle_cost / (2 * self.capacity)

    def info(self) -> str:
        return (
            f"{'-' * 25}\n"
            f"Battery {self.id} \n"
            f"Allowed Energy [{self.energy_min}-{self.energy_max}] \n"
            f"Energy Beginning/End [{self.energy_start}-{self.energy_end}] \n"
        )

    def __repr__(self) -> str:
        return f"Battery {self.id}"

    def is_connected(self) -> bool:
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
