import pytest

from battery_management.assets.battery import Battery


def test_battery_creation(sample_battery):
    assert isinstance(sample_battery, Battery)
    assert sample_battery.id == 42
    assert sample_battery.energy_start == 12
    assert sample_battery.energy_end == 40
    assert sample_battery.capacity == 40


def test_battery_energy_validation():
    with pytest.raises(ValueError):
        Battery(
            id=1,
            energy_start=50,
            energy_end=40,
            capacity=40,
            energy_min=5,
            energy_max=40,
            power_charge_max=5,
            power_discharge_max=5,
            connected=[True] * 30,
        )


def test_battery_info(sample_battery):
    info = sample_battery.info()
    assert "Battery 42" in info
    assert "Allowed Energy [5-40]" in info
    assert "Energy Beginning/End [12-40]" in info


def test_battery_is_connected(sample_battery):
    assert sample_battery.is_connected()


def test_battery_has_single_charging_session(sample_battery):
    assert sample_battery.has_single_charging_session()


def test_battery_add_cycle_costs(sample_battery):
    sample_battery.add_cycle_costs(battery_costs=1000, cycle_life=5000)
    assert sample_battery.cycle_cost == 0.2
    assert sample_battery.cycle_cost_per_kwh == 0.2 / (2 * sample_battery.capacity)


def test_battery_power_settings_validation():
    with pytest.raises(ValueError):
        Battery(
            id=1,
            energy_start=10,
            energy_end=20,
            capacity=40,
            energy_min=5,
            energy_max=40,
            power_charge_max=5,
            power_discharge_max=5,
            power_charge_min=1,
            connected=[True] * 30,
        )
