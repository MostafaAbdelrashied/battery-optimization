import pytest

from battery_management.assets.charging_point import ChargingPoint
from battery_management.assets.grid import Grid
from battery_management.assets.site import Site
from battery_management.assets.stationary_battery import StationaryBattery


def test_site_creation(sample_site):
    assert isinstance(sample_site, Site)
    assert sample_site.site_id == 1
    assert sample_site.n_charging_points == 2
    assert sample_site.country == "Germany"
    assert sample_site.voltage_level == 400


def test_site_charging_points(sample_site):
    assert len(sample_site.charging_points) == 1
    assert isinstance(sample_site.charging_points[0], ChargingPoint)
    assert sample_site.charging_points[0].asset_id == 1741


def test_site_stationary_batteries(sample_site):
    assert len(sample_site.stationary_batteries) == 1
    assert isinstance(sample_site.stationary_batteries[0], StationaryBattery)
    assert sample_site.stationary_batteries[0].id == 42


def test_site_grid(sample_site):
    assert isinstance(sample_site.grid, Grid)
    assert sample_site.grid.feed_power_limit == 100
    assert sample_site.grid.purchase_power_limit == 200


def test_site_load_restrictions(sample_site):
    assert sample_site.siteload_restriction_half_hour_charge == 150
    assert sample_site.siteload_restriction_half_hour_discharge == 100


def test_site_load_components(sample_site):
    assert sample_site.site_load_components == ["component1", "component2"]


def test_site_repr_and_str(sample_site):
    assert repr(sample_site) == "Site 1"
    expected_str = (
        "Site 1\n"
        "- Stationary Batteries: [StationaryBattery(id=42, capacity=100)]\n"
        "- Charging Points: [ChargingPoint(1741)]"
    )
    assert str(sample_site) == expected_str


def test_site_invalid_values():
    with pytest.raises(ValueError):
        Site(site_id=-1)

    with pytest.raises(ValueError):
        Site(site_id=1, n_charging_points=-2)


def test_site_add_charging_point(sample_site):
    new_charging_point = ChargingPoint(
        asset_id=1742,
        charging_power_kw=65,
        discharging_power_kw=60,
        expected_charging_efficiency=0.95,
        expected_discharging_efficiency=0.95,
    )

    sample_site.add_charging_point(new_charging_point)
    assert len(sample_site.charging_points) == 2
    assert sample_site.charging_points[-1].asset_id == 1742
    assert sample_site.n_charging_points == 3


def test_site_add_stationary_battery(sample_site):
    new_battery = StationaryBattery(
        id=43,
        capacity=200,
        energy_min=20,
        energy_max=180,
        power_charge_max=100,
        power_discharge_max=100,
    )
    sample_site.add_stationary_battery(new_battery)
    assert len(sample_site.stationary_batteries) == 2
    assert sample_site.stationary_batteries[-1].id == 43
