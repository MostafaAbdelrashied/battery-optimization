import pytest

from battery_management.assets.grid import Grid


def test_grid_creation(sample_grid):
    assert isinstance(sample_grid, Grid)
    assert sample_grid.feed_power_limit == 100
    assert sample_grid.purchase_power_limit == 200
    assert sample_grid.purchase_efficiency == 1  # Default value
    assert sample_grid.feed_efficiency == 0.9


def test_grid_dict_method(sample_grid):
    grid_dict = sample_grid.dict()
    assert isinstance(grid_dict, dict)
    assert grid_dict == {
        "feed_power_limit": 100,
        "purchase_power_limit": 200,
        "purchase_efficiency": 1,
        "feed_efficiency": 0.9,
    }


def test_grid_invalid_values():
    with pytest.raises(ValueError):
        Grid(feed_power_limit=-100, purchase_power_limit=200)

    with pytest.raises(ValueError):
        Grid(feed_power_limit=100, purchase_power_limit=-200)

    with pytest.raises(ValueError):
        Grid(feed_power_limit=100, purchase_power_limit=200, feed_efficiency=1.1)


def test_grid_str_representation(sample_grid):
    expected_str = "Grid(feed_power_limit=100, purchase_power_limit=200, feed_efficiency=0.9, purchase_efficiency=1.0)"
    assert str(sample_grid) == expected_str
