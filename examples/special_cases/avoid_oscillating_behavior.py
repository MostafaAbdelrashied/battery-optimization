# External Dependencies
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Package Dependencies
from battery_management.assets.battery import Battery
from battery_management.battery_optimizer import create

np.random.seed(42)
plt.style.use("ggplot")
# ----------------------------------------------------
# --- Example how to avoid unnecessary oscillation ---
# ----------------------------------------------------

# Define Sinus shaped Prices (this is quite accurate for a 24h cycle)
x = np.linspace(0, 29, 30)
prices = np.where(x < 14, 1, 0.1)
noise = 1 + np.random.randn(30) / 1000
prices *= noise

# Define two batteries along the V2G parameters
battery1 = Battery(
    **{
        "id": 42,
        "energy_start": 10,
        "energy_end": 40,
        "energy_min": 5,
        "energy_max": 40,
        "power_charge_max": 10,
        "power_discharge_max": 10,
        "connected": [True] * 30,
        "capacity": 40,
        "efficiency_charge": 1.0,
        "efficiency_discharge": 1.0,
    }
)
battery1.add_cycle_costs(battery_costs=10000)

# Instantiate the Optimizer
# we use dt = 0.5 <=> 1 time step is 30min
# - Add batteries
# - Add Prices
fo = create(id=42, dt=0.5)
_ = [fo.add_battery(battery) for battery in [battery1]]
fo.add_prices(prices, prices)

# Use the build-in functionality to create a plot
file_name = Path(__file__).parents[2] / Path(
    "figures/oscillation_behaviour/without_penalty.png"
)
file_name.parents[0].mkdir(parents=True, exist_ok=True)

# First Optimization: Prices only
res = fo.optimize()
fo.plot(res, file_name)

# ---------------------------
#  Step 2: apply the penalty
# ---------------------------
fo.include_battery_costs = True

file_name = Path(__file__).parents[2] / Path(
    "figures/oscillation_behaviour/with_penalty.png"
)
file_name.parents[0].mkdir(parents=True, exist_ok=True)

res = fo.optimize()
fo.plot(res, file_name)
