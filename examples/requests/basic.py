import pandas as pd
from battery_management.request_handler.request_handler import RequestHandler

test_request = {
    "id": 42,
    "start_time": pd.Timestamp(2020, 8, 31, 10),
    "end_time": pd.Timestamp(2020, 8, 31, 10),
}
test_request = {"request": test_request}

req1 = RequestHandler(test_request)

# ---- Stationary Battery ------
stationary_battery = dict(
    id=42,
    energy_min=5,
    energy_max=40,
    power_charge_max=5,
    power_discharge_max=5,
    capacity=40,
)
site1 = {"site_id": 12, "stationary_batteries": [stationary_battery]}
test_request["site_specifications"] = [site1]

# ---- Asset Status ------
as_stat = {
    "asset_id": 42,
    "soc_current_perc": 0.8,
    "soc_target_perc": 0,
    "battery_capacity_kwh": 100,
}
test_request["current_asset_status"] = [as_stat]

# ---- Request Handler ----
req2 = RequestHandler(test_request)
