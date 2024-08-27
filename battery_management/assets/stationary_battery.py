from dataclasses import dataclass, field
from typing import List

from battery_management.assets.battery import Battery


@dataclass
class StationaryBattery(Battery):
    """
    Represents a stationary battery, which is a special case of a Battery.
    """

    stationary: bool = field(default=True, init=False)
    connected: List[bool] = field(default_factory=lambda: [True], init=False)

    def __init__(
        self,
        id: int,
        capacity: float,
        energy_min: float,
        energy_max: float,
        power_charge_max: float,
        power_discharge_max: float,
        connected: List[bool] = None,
        **kwargs,
    ):
        super().__init__(
            id=id,
            capacity=capacity,
            energy_min=energy_min,
            energy_max=energy_max,
            power_charge_max=power_charge_max,
            power_discharge_max=power_discharge_max,
            connected=connected if connected is not None else [True],
            stationary=True,
            **kwargs,
        )

    def __repr__(self) -> str:
        return f"StationaryBattery(id={self.id}, capacity={self.capacity})"

    def __str__(self) -> str:
        return (
            f"Stationary Battery {self.id}\n"
            f"- Capacity: {self.capacity} kWh\n"
            f"- Max Power Charge: {self.power_charge_max} kW\n"
            f"- Max Power Discharge: {self.power_discharge_max} kW"
        )


if __name__ == "__main__":
    battery = StationaryBattery(
        id=42,
        energy_min=5,
        energy_max=40,
        capacity=40,
        power_charge_max=5,
        power_discharge_max=5,
    )

    print(battery)
    print([battery])
