# tests/fixtures/asset_status_fixture.py
import pytest

from battery_management.assets.asset_status import AssetStatus
from battery_management.assets.battery import Battery
from battery_management.assets.charging_point import ChargingPoint
from battery_management.assets.grid import Grid
from battery_management.assets.stationary_battery import StationaryBattery
from battery_management.assets.site import Site


@pytest.fixture
def sample_asset_dict():
    return {
        "asset_id": 42,
        "soc_current_perc": 0.8,
        "soc_target_perc": 0,
        "battery_capacity_kwh": 100,
    }

@pytest.fixture
def sample_asset_status(sample_asset_dict):
    return AssetStatus.from_dict(sample_asset_dict)



@pytest.fixture
def sample_battery():
    return Battery(
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

@pytest.fixture
def sample_charging_point_dict():
    return {
        "asset_id": 1741,
        "charging_power_kw": 65,
        "discharging_power_kw": 60,
        "expected_charging_efficiency": 0.95,
        "expected_discharging_efficiency": 0.95,
    }

@pytest.fixture
def sample_charging_point(sample_charging_point_dict):
    return ChargingPoint.from_dict(sample_charging_point_dict)



@pytest.fixture
def sample_grid():
    return Grid(feed_power_limit=100, purchase_power_limit=200, feed_efficiency=0.9)


@pytest.fixture
def sample_site():
    grid = Grid(feed_power_limit=100, purchase_power_limit=200, feed_efficiency=0.9)
    charging_point = ChargingPoint(asset_id=1741, charging_power_kw=65, discharging_power_kw=60, expected_charging_efficiency=0.95, expected_discharging_efficiency=0.95)
    stationary_battery = StationaryBattery(id=42, capacity=100, energy_min=5, energy_max=40, power_charge_max=5, power_discharge_max=5)
    
    return Site(
        site_id=1,
        n_charging_points=2,
        country="Germany",
        voltage_level=400,
        charging_points=[charging_point],
        stationary_batteries=[stationary_battery],
        grid=grid,
        siteload_restriction_half_hour_charge=150,
        siteload_restriction_half_hour_discharge=100,
        site_load_components=["component1", "component2"]
    )

@pytest.fixture
def sample_stationary_battery():
    return StationaryBattery(
        id=42,
        energy_min=5,
        energy_max=40,
        capacity=40,
        power_charge_max=5,
        power_discharge_max=5,
    )