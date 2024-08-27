from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from battery_management.assets.battery import Battery
from battery_management.battery_optimizer import create

np.random.seed(42)
plt.style.use("ggplot")

"""
For the Flex EV project we consider a charging park for electro-vehicles with a large on-site battery and PV 
installation. The Flex EV use-case is different from V2G in more than one aspect despite the superficial similarities. 
1) While there are electro-vehicles being charged, we treat these simply as an external sink of energy. The only
    relevant battery is the big one on-premise. 
2) We are considering a photo-voltaic installation on-site. Therefore the generalised site load can be both positive 
    and negative. This has some impact on the inner workings of the optimisation.
"""

# ------------ Simulate Problem --------------
time_steps = 2 * 24 * 4  # German energy market goes in 15min steps, we predict two days

# Define Sinus shaped Prices (this is quite accurate for a 24h cycle)
x = np.linspace(0, 4 * np.pi, time_steps)
x -= np.pi / 2
prices_import = np.sin(x) + 3
prices_export = (
    prices_import - 2
)  # Make sure that exporting into the grid is always more expensive

# Simulate something akin to PV production
x = np.linspace(0, np.pi, time_steps // 4)
daylight = list(np.sin(x))
night = [0] * (time_steps // 4)
half_night = [0] * (time_steps // 8)
# Let's start at midnight more or less
pv = (
    np.array(half_night + daylight + night + daylight + half_night) * 25
)  # PV max power in Duisburg is 25kW

# Let's try to simulate the energy consumption
# This sort of kind of follows a similar pattern as the price but with more randomness and zeroes followed by non-zeros
x = np.linspace(0, 4 * np.pi, time_steps)
x -= np.pi / 2
charging = np.sin(x) + np.random.randn(time_steps)
charging = np.where(charging > 0, charging, 0)
charging *= 30  # Typically high values would be around 30 kW

# Let's be careful! In the convention we use positive site load is consumption on site, negative site load production
site_load = charging - pv

# ------------ Optimizer --------------

# Define two batteries along the V2G parameters
battery = Battery(
    **{
        "id": 42,
        "capacity": 40,
        "energy_start": 10,
        "energy_end": 0,
        "energy_min": 5,
        "energy_max": 210,
        "power_charge_min": 0,
        "power_charge_max": 350,
        "power_discharge_max": 350,
        "connected": [True] * time_steps,
        "efficiency_charge": 1,
        "efficiency_discharge": 1,
    }
)

fo = create(id=42, type="OR", dt=0.5)
fo.calculate_savings = False
fo.allow_curtailment = True
fo.add_battery(battery)
fo.add_prices(prices_import, prices_export)

fo.add_site_load(site_load)
fo.add_site_limits(site_load_restriction_discharge=0)


# First Optimization: Prices only
res = fo.optimize()
df = res.aggregated_results

print(df.tail())

df2 = pd.DataFrame(res.grid_results)

# ------------ Plotting ------------
fig = plt.figure(figsize=(24, 8))
n = 7

plt.subplot(4, 2, 1)
plt.plot(prices_import, label="Price Import")
plt.plot(prices_export, label="Price Export")
plt.legend()

plt.subplot(4, 2, 3)
plt.plot(pv, label="Photovoltaic")
plt.legend()

plt.subplot(4, 2, 5)
plt.plot(charging, label="EV Charging")
plt.legend()

plt.subplot(4, 2, 7)
plt.plot(charging - pv, label="Effective Site Load")
plt.legend()

plt.subplot(4, 2, 2)
df["energy_content_kwh"].plot(label="Battery Energy")
plt.legend()

plt.subplot(4, 2, 4)
df["power_kw_site"].plot(label="Battery Power")
plt.legend()

# TODO Repair below
# plt.subplot(4, 2, 6)
# df2['feed'].plot(label='Feed')
# df2['purchase'].plot(label='Purchase')
# df2['curtail'].plot(label='Curtail')
# plt.legend()


fig.tight_layout()
plt.savefig(f"{Path(__file__).parents[2]}/figures/flex_ev.png")
