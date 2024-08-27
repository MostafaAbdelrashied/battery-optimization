from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from battery_management.assets.battery import Battery
from battery_management.battery_optimizer import create

plt.style.use("ggplot")
np.random.seed(42)


# Define prices with a step function. However, prices should be irrelevant here
x = np.linspace(0, 19, 20)
prices = np.where(x < 9, 1, 0.1)

# Define two batteries along the V2G parameters
battery1 = Battery(
    **{
        "id": 1,
        "capacity": 40,
        "energy_start": 10,
        "energy_end": 40,
        "energy_min": 5,
        "energy_max": 40,
        "power_charge_max": 1,
        "power_discharge_max": 1,
        "connected": [True] * 20,
        # Use 100% efficiency to make steps easier to understand
        "efficiency_charge": 1.0,
        "efficiency_discharge": 1.0,
    }
)

battery2 = Battery(
    **{
        "id": 2,
        "capacity": 40,
        "energy_start": 10,
        "energy_end": 40,
        "energy_min": 5,
        "energy_max": 40,
        "power_charge_max": 1,
        "power_discharge_max": 1,
        "connected": [True] * 18 + [False] * 2,
        # Use 100% efficiency to make steps easier to understand
        "efficiency_charge": 0.9,
        "efficiency_discharge": 0.9,
    }
)

# Instantiate the Optimizer
# - we use dt = 0.5 <=> 1 time step is 30min
# - Add batteries
# - Add Prices
fo = create(id=42, dt=1, fully_charged_as_penalty=True, calculate_savings=True)
_ = [fo.add_battery(battery) for battery in [battery1, battery2]]
triad = np.array([0] * 9 + [1] * 2 + [0] * 9)
fo.add_prices(
    tariffs_import=prices,
    tariffs_export=prices,
    triad_tariffs_import=triad,
    triad_tariffs_export=triad,
)

# Optimize, resulting plot will be produced automatically
res = fo.optimize()

# Create a plot to visualize the results
file_name = Path(__file__).parents[2] / Path("figures/max_power.png")
file_name.parents[0].mkdir(parents=True, exist_ok=True)
fo.plot(res, file_name)
