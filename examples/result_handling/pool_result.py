import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from battery_management.assets.battery import Battery
from battery_management.battery_optimizer import create
from battery_management.results_handler.pool_result import PoolResult

np.random.seed(42)
plt.style.use("ggplot")

# Define Sinus shaped Prices (this is quite accurate for a 24h cycle)
x = np.linspace(0, 2 * np.pi, 30)
prices_import = np.sin(x)

date_range = pd.date_range(start=pd.Timestamp(2020, 1, 1), freq="1h", periods=30)

# ---------------
# Site 1
# ---------------

# Define two batteries along the V2G parameters
battery1 = Battery(
    **{
        "id": 42,
        "energy_start": 10,
        "energy_end": 40,
        "energy_min": 5,
        "energy_max": 40,
        "power_charge_max": 5,
        "power_discharge_max": 5,
        "connected": [True] * 30,
        "capacity": 40,
        "efficiency_charge": 0.8,
        "efficiency_discharge": 0.8,
    }
)

battery2 = Battery(
    **{
        "id": 23,
        "capacity": 40,
        "energy_start": 12,
        "energy_end": 40,
        "energy_min": 5,
        "energy_max": 40,
        "power_charge_max": 5,
        "power_discharge_max": 5,
        "connected": [False] * 5 + [True] * 25,
        "efficiency_charge": 0.8,
        "efficiency_discharge": 0.8,
    }
)

# Instantiate the Optimizer
# optimizer type: OR Tools
# we use dt = 0.5 <=> 1 time step is 30min
# - Add batteries
# - Add Prices

fo = create(id=42, dt=0.5, type="OR")
_ = [fo.add_battery(battery) for battery in [battery1, battery2]]
fo.add_prices(prices_import, prices_import)
fo.add_date_range(date_range=date_range)

# First Optimization: Prices only
result1 = fo.optimize()


# ---------------
# Site 2
# ---------------

# Define two batteries along the V2G parameters
battery3 = Battery(
    **{
        "id": 122,
        "energy_start": 15,
        "energy_end": 30,
        "energy_min": 5,
        "energy_max": 40,
        "power_charge_max": 10,
        "power_discharge_max": 10,
        "connected": [True] * 30,
        "capacity": 40,
        "efficiency_charge": 0.8,
        "efficiency_discharge": 0.8,
    }
)

# Instantiate the Optimizer
# optimizer type: OR Tools
# we use dt = 0.5 <=> 1 time step is 30min
# - Add batteries
# - Add Prices

fo = create(id=27, dt=0.5, type="OR")
_ = [fo.add_battery(battery) for battery in [battery3]]
fo.add_prices(prices_import, prices_import)
fo.add_date_range(date_range=date_range)

# First Optimization: Prices only
result2 = fo.optimize()


result_dict = {42: result1, 27: result2}

pool_result = PoolResult(result_dict)

print("BATTERY")
with pd.option_context("display.max_rows", None, "display.max_columns", None):
    print(pool_result.battery_results)

print("SITE")
with pd.option_context("display.max_rows", None, "display.max_columns", None):
    print(pool_result.site_results)

print("POOL")
with pd.option_context("display.max_rows", None, "display.max_columns", None):
    print(pool_result.pool_results)
