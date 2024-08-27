import pytest

from battery_management.assets.battery import Battery
from battery_management.assets.charging_point import ChargingPoint


@pytest.fixture
def battery():
    return Battery(
        id=42,
        energy_start=12,
        energy_end=40,
        capacity=40,
        energy_min=5,
        energy_max=40,
        power_charge_max=10,
        power_discharge_max=10,
        connected=[False] * 5 + [True] * 25,
    )


@pytest.fixture
def charging_point():
    return ChargingPoint(
        dict(asset_id=42, charging_power_kw=10, discharging_power_kw=0)
    )
