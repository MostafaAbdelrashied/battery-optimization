from battery_management.assets.battery import Battery


class StationaryBattery(Battery):
    def __init__(
        self,
        id: int,
        energy_min: float,
        energy_max: float,
        capacity: float,
        power_charge_max: float,
        power_discharge_max: float,
        power_charge_min: float = 0,
        efficiency_charge: float = 1.0,
        efficiency_discharge: float = 1.0,
        cycle_life: int = 5000,
        cycle_usage: int = 0,
        battery_costs: float = 0,
    ):
        super().__init__(
            id=id,
            energy_start=energy_min,
            energy_end=energy_min,
            energy_min=energy_min,
            energy_max=energy_max,
            power_charge_max=power_charge_max,
            power_discharge_max=power_discharge_max,
            connected=[True],
            capacity=capacity,
            power_charge_min=power_charge_min,
            efficiency_charge=efficiency_charge,
            efficiency_discharge=efficiency_discharge,
            stationary=True,
            cycle_life=cycle_life,
            cycle_usage=cycle_usage,
            battery_costs=battery_costs,
        )

    def __repr__(self) -> str:
        # StationaryBattery(id=42, capacity=100)"
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
