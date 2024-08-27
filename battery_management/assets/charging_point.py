from dataclasses import dataclass, field
from typing import Dict

import pandas as pd


@dataclass
class ChargingPoint:
    asset_id: int
    charging_power_kw: float
    discharging_power_kw: float = 0
    expected_charging_efficiency: float = 1.0
    expected_discharging_efficiency: float = 1.0
    connected_time_schedule: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __repr__(self) -> str:
        return f"ChargingPoint({self.asset_id})"

    def is_available(self, start: pd.Timestamp, end: pd.Timestamp) -> bool:
        if self.connected_time_schedule.empty:
            return True
        return not self.connected_time_schedule.loc[
            start : end - pd.Timedelta(seconds=1), "connected"
        ].any()

    def book(self, start: pd.Timestamp, end: pd.Timestamp) -> Dict[str, str]:
        if not self.is_available(start, end):
            return {
                "message": f"ChargingPoint {self.asset_id} already booked for this period."
            }

        new_schedule = pd.DataFrame(
            index=pd.date_range(start, end, freq="10min"), data={"connected": 1}
        )
        self.connected_time_schedule = (
            pd.concat([self.connected_time_schedule, new_schedule])
            .resample("10min")
            .max()
        )
        return {"message": "Booked."}

    def reset(self) -> None:
        self.connected_time_schedule = pd.DataFrame()

    @classmethod
    def from_dict(cls, charging_point: Dict) -> "ChargingPoint":
        return cls(
            asset_id=charging_point["asset_id"],
            charging_power_kw=charging_point["charging_power_kw"],
            discharging_power_kw=charging_point.get("discharging_power_kw", 0),
            expected_charging_efficiency=charging_point.get(
                "expected_charging_efficiency", 1
            ),
            expected_discharging_efficiency=charging_point.get(
                "expected_discharging_efficiency", 1
            ),
        )


if __name__ == "__main__":
    point1 = {
        "asset_id": 1741,
        "charging_power_kw": 65,
        "discharging_power_kw": 60,
        "expected_charging_efficiency": 0.95,
        "expected_discharging_efficiency": 0.95,
    }

    cp = ChargingPoint.from_dict(point1)
    print(cp)

    start_time = pd.Timestamp("2023-01-01 10:00:00")
    end_time = pd.Timestamp("2023-01-01 12:00:00")

    print(f"Is available: {cp.is_available(start_time, end_time)}")
    booking_result = cp.book(start_time, end_time)
    print(f"Booking result: {booking_result}")
    print(f"Is available after booking: {cp.is_available(start_time, end_time)}")

    cp.reset()
    print(f"Is available after reset: {cp.is_available(start_time, end_time)}")
