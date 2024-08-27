from typing import List

import numpy as np
import pandas as pd
from loguru import logger


class Battery:
    """
    Represents a battery with various parameters for optimization.

    Parameters
    ----------
    id : int
        The unique identifier of the battery.
    energy_start : float
        The initial energy level of the battery.
    energy_end : float
        The target energy level at the end of the optimization period.
    energy_min : float
        The minimum allowed energy level of the battery.
    energy_max : float
        The maximum allowed energy level of the battery.
    power_charge_max : float
        The maximum charging power of the battery.
    power_discharge_max : float
        The maximum discharging power of the battery.
    connected : List[bool]
        A list indicating when the battery is connected.
    capacity : float
        The total capacity of the battery.
    power_charge_min : float, optional
        The minimum charging power of the battery. Defaults to 0.
    efficiency_charge : float, optional
        The charging efficiency of the battery. Defaults to 1.0.
    efficiency_discharge : float, optional
        The discharging efficiency of the battery. Defaults to 1.0.
    stationary : bool, optional
        Whether the battery is stationary or not. Defaults to False.
    cycle_life : int, optional
        The expected cycle life of the battery. Defaults to 5000.
    cycle_usage : int, optional
        The current cycle usage of the battery. Defaults to 0.
    battery_costs : float, optional
        The cost of the battery. Defaults to 0.

    Attributes
    ----------
    id : int
    energy_min : float
    energy_max : float
    energy_start : float
    energy_end : float
    efficiency_charge : float
    efficiency_discharge : float
    capacity : float
    power_charge_max : float
    power_charge_min : float
    power_discharge_max : float
    connected : List[bool]
    affected_charging_point_id : Optional[int]
    stationary : bool
    cycle_life : int
    cycle_usage : int
    battery_costs : float
    cycle_cost : float
    cycle_cost_per_kwh : float
    """

    def __init__(
        self,
        id: int,
        energy_start: float,
        energy_end: float,
        energy_min: float,
        energy_max: float,
        power_charge_max: float,
        power_discharge_max: float,
        connected: List[bool],
        capacity: float,
        power_charge_min: float = 0,
        efficiency_charge: float = 1.0,
        efficiency_discharge: float = 1.0,
        stationary: bool = False,
        cycle_life: int = 5000,
        cycle_usage: int = 0,
        battery_costs: float = 0,
    ):
        self.id = id
        self.energy_min = energy_min
        self.energy_max = energy_max
        self._validate_energy(energy_start, "Energy at start")
        self.energy_start = energy_start
        self._validate_energy(energy_end, "Energy at end")
        self.energy_end = energy_end
        self.efficiency_charge = efficiency_charge
        self.efficiency_discharge = efficiency_discharge
        self.capacity = capacity
        self.power_charge_max = power_charge_max
        self.power_charge_min = power_charge_min
        self.power_discharge_max = power_discharge_max
        self.connected = connected
        self.affected_charging_point_id = None
        self.stationary = stationary

        assert power_discharge_max == 0 or power_charge_min == 0, (
            "Cannot set a value to power_charge_min if " "power_discharge_max is not 0."
        )

        self.cycle_life = cycle_life
        self.cycle_usage = cycle_usage
        self.battery_costs = battery_costs
        self.cycle_cost = self.battery_costs / self.cycle_life
        self.cycle_cost_per_kwh = self.cycle_cost / (2 * self.capacity)

    def _validate_energy(self, energy: float, description: str) -> None:
        """
        Validate that the given energy is within the battery limits.

        Parameters
        ----------
        energy : float
            The energy value to validate.
        description : str
            A description of the energy value for error messaging.
        """
        if energy < self.energy_min or energy > self.energy_max:
            logger.error(f"{description} outside of battery limits")

    def info(self) -> str:
        """
        Get a string representation of the battery's information.

        Returns
        -------
        str
            A formatted string containing the battery's information.
        """
        return (
            f"{'-' * 25}\n"
            f"Battery {self.id} \n"
            f"Allowed Energy [{self.energy_min}-{self.energy_max}] \n"
            f"Energy Beginning/End [{self.energy_start}-{self.energy_end}] \n"
        )

    def __repr__(self) -> str:
        return f"Battery {self.id}"

    def is_connected(self) -> bool:
        """
        Check if the battery is connected at any point.

        Returns
        -------
        bool
            True if the battery is connected at any point, False otherwise.
        """
        return np.int64(self.connected).sum() > 0

    def has_single_charging_session(self) -> bool:
        """
        Check if the battery has a single continuous charging session.

        Returns
        -------
        bool
            True if there's a single continuous charging session, False otherwise.
        """
        df = pd.DataFrame({"connected": np.int64(self.connected)})
        diffs = df["connected"].diff()

        if sum(diffs == 1) > 1 or sum(diffs == -1) > 1:
            return False

        if sum(diffs == -1) == 1 and sum(diffs == 1) == 1:
            start_idx = np.where(diffs == 1)[0][0]
            end_idx = np.where(diffs == -1)[0][0]
            return start_idx < end_idx

        return True

    def add_cycle_costs(self, battery_costs: float, cycle_life: int = 5000) -> None:
        """
        Add cycle costs to the battery.

        Parameters
        ----------
        battery_costs : float
            The cost of the battery.
        cycle_life : int, optional
            The expected cycle life of the battery. Defaults to 5000.
        """
        self.cycle_life = cycle_life
        self.battery_costs = battery_costs
        self.cycle_cost = self.battery_costs / self.cycle_life
        self.cycle_cost_per_kwh = self.cycle_cost / (2 * self.capacity)


if __name__ == "__main__":
    battery = Battery(
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

    print(battery.info())
    print(battery)
