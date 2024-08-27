import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from battery_management.assets.stationary_battery import StationaryBattery
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

df = pd.read_csv(f"{Path(__file__).parent}/data/optimization_testcase.csv")
prices = df["price_buy"].values
pv = df["pv"].values
charging = df["load"].values
site_load = charging - pv

time_steps = len(df)

# ------------ Optimizer --------------
# Define two batteries along the V2G parameters

# Determine cycle cost. The battery has a lifetime of 5000 cycles and costs 35 000 Euro. A full cycle corresponds to
# charging and discharging from 0.09 * 210 to 210 * 0.99 (i.e. 0.9 * 210 kWh) or 189kWh. So one cycle is 378 kWh and
# costs 7 Euro. The resulting penalty here is (7/378) Eur/kWh = 0.0185 Eur/kWh applied to the absolute value of any
# energy changes.

battery = StationaryBattery(
    **{
        "id": 42,
        "capacity": 40,
        "energy_start": 0.8202 * 210,
        "energy_end": 5,
        "energy_min": 5,
        "energy_max": 210,
        "power_charge_min": 0,
        "power_charge_max": 350,
        "power_discharge_max": 350,
        "connected": [True] * time_steps,
        "efficiency_charge": 1,
        "efficiency_discharge": 1,
        "cycle_life": 5000,
        "cycle_usage": 0,
        "battery_costs": 35000,
    }
)

fo = create(id=42, type="OR", dt=0.25)

# Activate Necessary Flags
fo.calculate_savings = False
fo.allow_curtailment = True
fo.include_battery_costs = True

fo.add_battery(battery)
fo.add_prices(prices, prices)

fo.add_site_load(site_load)
fo.add_site_limits(site_load_restriction_discharge=0)

# First Optimization: Prices only
res = fo.optimize()
result = res.aggregated_results

print(result.head(25))
print(result.tail(25))

print(res.objective_value)

print(df.head())
print(df.tail())

grid = pd.DataFrame(res.grid_results)

# ------------ Plotting ------------
plt.figure(figsize=(24, 8))

# 1) Compare Energy
plt.subplot(2, 2, 1)

plt.plot((210 * df["SoC"]).values, label="Specialised Optimizer")
result["energy_content_kwh"].plot(label="General Optimizer")
plt.legend(title="SOC [kWh]")

plt.subplot(2, 2, 2)
plt.plot(prices, label="Price Import")
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(pv, label="Photovoltaic")
plt.plot(charging, label="EV Charging")
plt.plot(charging - pv, label="Effective Site Load")
plt.legend()

plt.subplot(2, 2, 4)
df["site_delta_power"].plot(label="Site Delta Power")
plt.legend()

_path = f"{Path(__file__).parents[2]}/figures"
os.makedirs(_path, exist_ok=True)
plt.savefig(f"{_path}/flexev_comparison.png")

"""
Questions:
1) What are price_buy/sell? There seems to be a factor of 10 between this and the price file. Also, wrt actual cases,
    we need to be careful with MWh/kWh. I personally prefer kWh as it is more meaningful in the context we operate in
2) Is there a penalty used for the optimised SOC? In the period I randomly picked (2019/07/01-2019/07/02, 48h) 
    the SoC column remains constant throughout the entire period
3) site_delta_power can be pos/neg, what does this mean? Also, what does delta mean with respect to power?
4) 
"""
