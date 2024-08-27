from battery_management.assets.battery import Battery
from battery_management.assets.stationary_battery import StationaryBattery

def test_stationary_battery_creation(sample_stationary_battery):
    assert isinstance(sample_stationary_battery, StationaryBattery)
    assert sample_stationary_battery.id == 42
    assert sample_stationary_battery.energy_min == 5
    assert sample_stationary_battery.energy_max == 40
    assert sample_stationary_battery.capacity == 40
    assert sample_stationary_battery.power_charge_max == 5
    assert sample_stationary_battery.power_discharge_max == 5

def test_stationary_battery_default_values(sample_stationary_battery):
    assert sample_stationary_battery.stationary is True
    assert sample_stationary_battery.energy_start == sample_stationary_battery.energy_min
    assert sample_stationary_battery.energy_end == sample_stationary_battery.energy_min
    assert sample_stationary_battery.connected == [True]

def test_stationary_battery_repr_and_str(sample_stationary_battery):
    assert repr(sample_stationary_battery) == "StationaryBattery(id=42, capacity=40)"
    assert str(sample_stationary_battery) == (
        "Stationary Battery 42\n"
        "- Capacity: 40 kWh\n"
        "- Max Power Charge: 5 kW\n"
        "- Max Power Discharge: 5 kW"
    )

def test_stationary_battery_inheritance():
    assert issubclass(StationaryBattery, Battery)
