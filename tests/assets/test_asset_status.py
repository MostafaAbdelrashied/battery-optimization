from battery_management.assets.asset_status import AssetStatus

def test_asset_status_creation(sample_asset_status):
    assert isinstance(sample_asset_status, AssetStatus)
    assert sample_asset_status.asset_id == 42
    assert sample_asset_status.battery_capacity_kwh == 100
    assert sample_asset_status.soc_current_perc == 0.8
    assert sample_asset_status.soc_target_perc == 0

def test_asset_status_clamp_soc():
    as1 = AssetStatus(asset_id=1, battery_capacity_kwh=100, soc_current_perc=1.2)
    as2 = AssetStatus(asset_id=2, battery_capacity_kwh=100, soc_current_perc=-0.2)
    assert as1.soc_current_perc == 1
    assert as2.soc_current_perc == 0

def test_asset_status_repr_and_str(sample_asset_status):
    assert repr(sample_asset_status) == "Asset 42"
    assert str(sample_asset_status) == (
        "Asset 42\n"
        "- SOC Current [%]: 80.00%\n"
        "- SOC Target [%]: 0.00%\n"
        "- Capacity [kWh]: 100"
    )

def test_asset_status_from_dict(sample_asset_dict):
    as_from_dict = AssetStatus.from_dict(sample_asset_dict)
    assert isinstance(as_from_dict, AssetStatus)
    assert as_from_dict.asset_id == 42
    assert as_from_dict.battery_capacity_kwh == 100
    assert as_from_dict.soc_current_perc == 0.8
    assert as_from_dict.soc_target_perc == 0

