import pandas as pd


def test_is_available(charging_point):
    start = pd.Timestamp(2020, 1, 1)
    end = pd.Timestamp(2020, 1, 2)
    assert charging_point.is_available(start, end)


def test_book(charging_point):
    start = pd.Timestamp(2020, 1, 1, 10)
    end = pd.Timestamp(2020, 1, 1, 16)

    out = charging_point.book(start, end)

    assert not charging_point.is_available(start, end)
    assert not charging_point.is_available(
        start + pd.Timedelta("4h"), end + pd.Timedelta("4h")
    )
    assert charging_point.is_available(start - pd.Timedelta("4h"), start)
    assert out["message"] == "Booked."

    start = pd.Timestamp(2020, 1, 1, 12)
    end = pd.Timestamp(2020, 1, 2, 12)

    out = charging_point.book(start, end)
    assert out["message"] != "Booked."

    # Testing the edges
    start = pd.Timestamp(2020, 1, 1, 9)
    end = pd.Timestamp(2020, 1, 1, 10)
    out = charging_point.book(start, end)
    assert out["message"] == "Booked."

    start = pd.Timestamp(2020, 1, 1, 16)
    end = pd.Timestamp(2020, 1, 1, 20)
    out = charging_point.book(start, end)
    assert out["message"] == "Booked."

    # Looking for gaps
    start = pd.Timestamp(2020, 1, 1, 9)
    end = pd.Timestamp(2020, 1, 1, 20) - pd.Timedelta(seconds=1)
    print(charging_point.connected_time_schedule.loc[start:end, "connected"])
    assert charging_point.connected_time_schedule.loc[start:end, "connected"].min() == 1
