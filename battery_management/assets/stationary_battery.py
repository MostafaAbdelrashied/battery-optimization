from typing import Any, Dict

from battery_management.assets.battery import Battery


class StationaryBattery(Battery):
    """
    Represents a stationary battery, which is a special case of a Battery.

    This class inherits from the Battery class and sets some default values
    specific to stationary batteries.

    Parameters
    ----------
    **kwargs : Dict[str, Any]
        Keyword arguments to initialize the battery. See Battery class for details.

    Attributes
    ----------
    All attributes are inherited from the Battery class.
    """

    def __init__(self, **kwargs: Dict[str, Any]):
        # Set default values for stationary batteries
        kwargs.setdefault("energy_start", kwargs.get("energy_min"))
        kwargs.setdefault("energy_end", kwargs.get("energy_min"))
        kwargs.setdefault("connected", [True])

        super().__init__(**kwargs)
        self.stationary = True

    def __repr__(self) -> str:
        return f"Stat.Bat.{self.id}"

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
