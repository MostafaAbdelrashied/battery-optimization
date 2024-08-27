from dataclasses import dataclass
from typing import Optional


@dataclass
class AssetStatus:
    asset_id: int
    battery_capacity_kwh: float
    soc_current_perc: Optional[float] = None
    soc_target_perc: Optional[float] = None
    power_current_kw: Optional[float] = None
    charge_limit_kw: Optional[float] = None
    discharge_limit_kw: Optional[float] = None

    def __post_init__(self):
        self.soc_current_perc = self._clamp_soc(self.soc_current_perc)
        self.soc_target_perc = self._clamp_soc(self.soc_target_perc)

    @staticmethod
    def _clamp_soc(value: Optional[float]) -> Optional[float]:
        if value is not None:
            return max(0, min(value, 1))
        return None

    def __repr__(self) -> str:
        return f"Asset {self.asset_id}"

    def __str__(self) -> str:
        return (
            f"Asset {self.asset_id}\n"
            f"- SOC Current [%]: {self.soc_current_perc:.2%}\n"
            f"- SOC Target [%]: {self.soc_target_perc:.2%}\n"
            f"- Capacity [kWh]: {self.battery_capacity_kwh}"
        )

    @classmethod
    def from_dict(cls, asset: dict) -> "AssetStatus":
        return cls(
            asset_id=asset["asset_id"],
            battery_capacity_kwh=asset["battery_capacity_kwh"],
            soc_current_perc=asset.get("soc_current_perc"),
            soc_target_perc=asset.get("soc_target_perc"),
            power_current_kw=asset.get("power_current_kw"),
            charge_limit_kw=asset.get("charge_limit_kw"),
            discharge_limit_kw=asset.get("discharge_limit_kw"),
        )


if __name__ == "__main__":
    as_stat = {
        "asset_id": 42,
        "soc_current_perc": 0.8,
        "soc_target_perc": 0,
        "battery_capacity_kwh": 100,
    }
    asset_status = AssetStatus.from_dict(as_stat)
    print(asset_status)
    print([asset_status])
