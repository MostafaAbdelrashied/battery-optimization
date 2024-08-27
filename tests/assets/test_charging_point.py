import pandas as pd
from battery_management.assets.charging_point import ChargingPoint

def test_charging_point_creation(sample_charging_point):
    assert isinstance(sample_charging_point, ChargingPoint)
    assert sample_charging_point.asset_id == 1741
    assert sample_charging_point.charging_power_kw == 65
    assert sample_charging_point.discharging_power_kw == 60

def test_charging_point_availability(sample_charging_point):
    start = pd.Timestamp("2023-01-01 10:00:00")
    end = pd.Timestamp("2023-01-01 12:00:00")
    assert sample_charging_point.is_available(start, end)

def test_charging_point_booking(sample_charging_point):
    start = pd.Timestamp("2023-01-01 10:00:00")
    end = pd.Timestamp("2023-01-01 12:00:00")
    booking_result = sample_charging_point.book(start, end)
    assert booking_result["message"] == "Booked."
    assert not sample_charging_point.is_available(start, end)

def test_charging_point_reset(sample_charging_point):
    start = pd.Timestamp("2023-01-01 10:00:00")
    end = pd.Timestamp("2023-01-01 12:00:00")
    sample_charging_point.book(start, end)
    sample_charging_point.reset()
    assert sample_charging_point.is_available(start, end)

def test_charging_point_from_dict(sample_charging_point_dict):
    cp = ChargingPoint.from_dict(sample_charging_point_dict)
    assert isinstance(cp, ChargingPoint)
    assert cp.asset_id == 1741
    assert cp.charging_power_kw == 65
    assert cp.discharging_power_kw == 60