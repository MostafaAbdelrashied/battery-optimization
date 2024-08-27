from typing import Any, Dict

import pandas as pd
from loguru import logger


class ChargingPoint:
    """
    Represents a single charging point with various attributes and booking functionality.

    Parameters
    ----------
    charging_point : Dict[str, Any]
        A dictionary containing charging point information.

    Attributes
    ----------
    asset_id : Any
        The unique identifier of the charging point.
    charging_power_kw : float
        The maximum charging power in kW.
    discharging_power_kw : float
        The maximum discharging power in kW.
    expected_charging_efficiency : float
        The expected charging efficiency.
    expected_discharging_efficiency : float
        The expected discharging efficiency.
    connected_time_schedule : pd.DataFrame
        A DataFrame representing the charging point's connection schedule.
    """

    def __init__(self, charging_point: Dict[str, Any]):
        self.asset_id = charging_point["asset_id"]
        self.charging_power_kw = charging_point["charging_power_kw"]
        self.discharging_power_kw = charging_point.get("discharging_power_kw", 0)
        self.expected_charging_efficiency = charging_point.get(
            "expected_charging_efficiency", 1
        )
        self.expected_discharging_efficiency = charging_point.get(
            "expected_discharging_efficiency", 1
        )
        self.connected_time_schedule = pd.DataFrame()

    def __repr__(self) -> str:
        return f"ChargingPoint({self.asset_id})"

    def is_available(self, start: pd.Timestamp, end: pd.Timestamp) -> bool:
        """
        Check if the charging point is available during the specified time period.

        Parameters
        ----------
        start : pd.Timestamp
            The start time of the period to check.
        end : pd.Timestamp
            The end time of the period to check.

        Returns
        -------
        bool
            True if the charging point is available, False otherwise.
        """
        if self.connected_time_schedule.empty:
            return True

        return (
            self.connected_time_schedule.loc[
                start : end - pd.Timedelta(seconds=1), "connected"
            ].sum()
            == 0
        )

    def book(self, start: pd.Timestamp, end: pd.Timestamp) -> Dict[str, str]:
        """
        Book the charging point for the specified time period.

        Parameters
        ----------
        start : pd.Timestamp
            The start time of the booking.
        end : pd.Timestamp
            The end time of the booking.

        Returns
        -------
        Dict[str, str]
            A dictionary containing a message about the booking status.
        """
        if not self.is_available(start, end):
            message = f"ChargingPoint {self.asset_id} already booked for this period."
            logger.info(message)
            return {"message": message}

        start_index = pd.Timestamp(start.value).replace(
            second=0, microsecond=0
        ) - pd.Timedelta(minutes=1)
        end_index = pd.Timestamp(end.value).replace(
            second=0, microsecond=0
        ) + pd.Timedelta(minutes=1)

        index = pd.date_range(start_index.value, end_index.value, freq="10min")
        booking = pd.DataFrame(index=index, data={"connected": 0})
        booking.loc[start : end - pd.Timedelta(seconds=1), "connected"] = 1

        self.connected_time_schedule = (
            pd.concat([self.connected_time_schedule, booking]).resample("10min").sum()
        )
        assert self.connected_time_schedule["connected"].max() <= 1

        return {"message": "Booked."}

    def reset(self) -> None:
        """
        Reset the charging point's connection schedule.
        """
        self.connected_time_schedule = pd.DataFrame()


if __name__ == "__main__":
    point1 = {
        "asset_id": 1741,
        "charging_power_kw": 65,
        "discharging_power_kw": 60,
        "expected_charging_efficiency": 0.95,
        "expected_discharging_efficiency": 0.95,
    }

    cp = ChargingPoint(point1)
    print(cp)

    # Example usage
    start_time = pd.Timestamp("2023-01-01 10:00:00")
    end_time = pd.Timestamp("2023-01-01 12:00:00")

    print(f"Is available: {cp.is_available(start_time, end_time)}")
    booking_result = cp.book(start_time, end_time)
    print(f"Booking result: {booking_result}")
    print(f"Is available after booking: {cp.is_available(start_time, end_time)}")

    cp.reset()
    print(f"Is available after reset: {cp.is_available(start_time, end_time)}")
