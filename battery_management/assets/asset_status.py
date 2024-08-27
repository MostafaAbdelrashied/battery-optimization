from typing import Dict, Optional


class AssetStatus:
    """
    Represents the status of an asset with various parameters.

    Parameters
    ----------
    asset : Dict
        A dictionary containing asset information.

    Attributes
    ----------
    asset_id : int
        The unique identifier of the asset.
    soc_current_perc : Optional[float]
        The current state of charge percentage.
    soc_target_perc : Optional[float]
        The target state of charge percentage.
    battery_capacity_kwh : float
        The battery capacity in kWh.
    power_current_kw : Optional[float]
        The current power in kW.
    charge_limit_kw : Optional[float]
        The charging power limit in kW.
    discharge_limit_kw : Optional[float]
        The discharging power limit in kW.
    """

    def __init__(self, asset: Dict):
        self.asset_id: int = asset["asset_id"]
        self.soc_current_perc: Optional[float] = self._clamp_soc(
            asset.get("soc_current_perc")
        )
        self.soc_target_perc: Optional[float] = self._clamp_soc(
            asset.get("soc_target_perc")
        )
        self.battery_capacity_kwh: float = asset["battery_capacity_kwh"]
        self.power_current_kw: Optional[float] = asset.get("power_current_kw")
        self.charge_limit_kw: Optional[float] = asset.get("charge_limit_kw")
        self.discharge_limit_kw: Optional[float] = asset.get("discharge_limit_kw")

    @staticmethod
    def _clamp_soc(value: Optional[float]) -> Optional[float]:
        """
        Clamp the state of charge value between 0 and 1.

        Parameters
        ----------
        value : Optional[float]
            The value to be clamped.

        Returns
        -------
        Optional[float]
            The clamped value, or None if the input was None.
        """
        if value is not None:
            return max(0, min(value, 1))
        return None

    def __repr__(self) -> str:
        return f"Asset {self.asset_id}"

    def __str__(self) -> str:
        return (
            f"Asset {self.asset_id}\n"
            f"- SOC Current [%]: {self.soc_current_perc}\n"
            f"- SOC Target [%]: {self.soc_target_perc}\n"
            f"- Capacity [kWh]: {self.battery_capacity_kwh}"
        )


if __name__ == "__main__":
    as_stat = {
        "asset_id": 42,
        "soc_current_perc": 0.8,
        "soc_target_perc": 0,
        "battery_capacity_kwh": 100,
    }
    print(AssetStatus(as_stat))
    print([AssetStatus(as_stat)])
