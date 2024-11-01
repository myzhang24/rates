import datetime as dt
import numpy as np
import pandas as pd

import sys
import inspect
import time
import logging
logging.basicConfig(level=logging.INFO)

########################################################################################################################
# Date and schedule tests
########################################################################################################################

def debug_sifma_holidays():
    from date_util import _SIFMA_
    # Check if a date is a business day
    test_date = dt.datetime(2024, 6, 19)
    assert not _SIFMA_.is_biz_day(test_date)
    assert _SIFMA_.prev_biz_day(test_date) == dt.datetime(2024, 6, 18)
    assert _SIFMA_.next_biz_day(test_date) == dt.datetime(2024, 6, 20)

    # Example usage
    st = dt.datetime(2018, 4, 1)
    et = dt.datetime(2024, 10, 9)
    days = pd.date_range(st, et, freq='D')
    days = [x for x in days if not _SIFMA_.is_biz_day(x)]
    assert len(days) == 753


def debug_fomc_generation():
    from date_util import generate_fomc_meeting_dates
    st = dt.datetime(2018, 4, 1)
    et = dt.datetime(2024, 10, 9)
    fomc_meetings = generate_fomc_meeting_dates(st, et)
    assert len(fomc_meetings) == 52


def debug_date_conversion():
    from date_util import parse_date, convert_date
    st = dt.datetime(2018, 4, 1)
    assert convert_date(st) == 41729
    assert parse_date(41729) == st

    dates = pd.date_range("2020-01-01", "2020-02-01", freq="d")
    int_array = convert_date(dates)
    assert isinstance(int_array, np.ndarray)
    assert int_array.sum() == 1356304

    dates = parse_date(int_array)
    assert isinstance(dates, np.ndarray)
    assert dates[0] == dt.datetime(2020, 1, 1)


########################################################################################################################
# Curve tests
########################################################################################################################
ff_prices = pd.Series({
    "FFV4": 95.1725,
    "FFX4": 95.33,
    "FFZ4": 95.48,
    "FFF5": 95.64,
    "FFG5": 95.815,
    "FFH5": 95.89,
    "FFJ5": 96.015,
    "FFK5": 96.125,
    "FFM5": 96.21,
    "FFN5": 96.30,
    "FFQ5": 96.385,
    "FFU5": 96.425,
    "FFV5": 96.475
})
sofr_1m_prices = pd.Series({
    "SERV4": 95.1525,
    "SERX4": 95.310,
    "SERZ4": 95.435,
    "SERF5": 95.595,
    "SERG5": 95.785,
    "SERH5": 95.860,
    "SERJ5": 95.985,
    "SERK5": 96.095,
    "SERM5": 96.175,
    "SERN5": 96.265,
    "SERQ5": 96.345,
    "SERU5": 96.375,
    "SERV5": 96.425
}, name="SOFR1M")
sofr_3m_prices = pd.Series({
    "SFRU4": 95.2075,
    "SFRZ4": 95.660,
    "SFRH5": 96.025,
    "SFRM5": 96.285,
    "SFRU5": 96.45,
    "SFRZ5": 96.550,
    "SFRH6": 96.600,
    "SFRM6": 96.62,
    "SFRU6": 96.620,
    "SFRZ6": 96.610,
    "SFRH7": 96.605,
    "SFRM7": 96.595,
    "SFRU7": 96.590,
    "SFRZ7": 96.575,
    "SFRH8": 96.560,
    "SFRM8": 96.545,
}, name="SOFR3M")
sofr_swaps_rates = pd.Series({
    "1W": 4.8400,
    "2W": 4.84318,
    "3W": 4.8455,
    "1M": 4.8249,
    "2M": 4.7530,
    "3M": 4.6709,
    "4M": 4.6020,
    "5M": 4.5405,
    "6M": 4.4717,
    "7M": 4.41422,
    "8M": 4.35880,
    "9M": 4.3061,
    "10M": 4.2563,
    "11M": 4.2110,
    "12M": 4.16675,
    "18M": 3.9378,
    "2Y": 3.81955,
    "3Y": 3.6866,
    "4Y": 3.61725,
    "5Y": 3.5842,
    "6Y": 3.5735,
    "7Y": 3.5719,
    "10Y": 3.5972,
    "15Y": 3.6590,
    "20Y": 3.6614,
    "30Y": 3.51965
})
sofr_swaps_long = pd.Series({
    "1Y": 4.16675,
    "2Y": 3.81955,
    "3Y": 3.6866,
    "4Y": 3.61725,
    "5Y": 3.5842,
    "6Y": 3.5735,
    "7Y": 3.5719,
    "10Y": 3.5972,
    "15Y": 3.6590,
    "20Y": 3.6614,
    "25Y": 3.5984,
    "30Y": 3.51965
})

def debug_ff_calibration():
    from curve import USDCurve
    # FF
    ff = USDCurve("FF", "2024-10-09")
    ff.calibrate_future_curve(ff_prices, on_penalty=False)
    err = 1e2 * (ff.price_1m_futures(ff_prices.index) - ff_prices.values)
    assert np.abs(err).sum() < 2

def debug_sofr1m_calibration():
    from curve import USDCurve
    # SOFR1M
    sofr1m = USDCurve("SOFR", "2024-10-09")
    sofr1m.calibrate_future_curve(sofr_1m_prices)
    err = 1e2 * (sofr1m.price_1m_futures(sofr_1m_prices.index) - sofr_1m_prices.values)
    assert np.abs(err).sum() < 4

def debug_sofr3m_calibration():
    from curve import USDCurve
    # SOFR3M
    sofr3m = USDCurve("SOFR", "2024-10-09")
    sofr3m.calibrate_future_curve(futures_3m_prices=sofr_3m_prices)
    err = 1e2 * (sofr3m.price_3m_futures(sofr_3m_prices.index) - sofr_3m_prices.values)
    assert np.abs(err).sum() < 2

def debug_sofr_joint_calibration():
    from curve import USDCurve
    # SOFR1M and 3M
    sofr = USDCurve("SOFR", "2024-10-09")
    sofr.calibrate_future_curve(sofr_1m_prices, sofr_3m_prices)
    err = 1e2 * (sofr.price_1m_futures(sofr_1m_prices.index) - sofr_1m_prices.values)
    err3 = 1e2 * (sofr.price_3m_futures(sofr_3m_prices.index) - sofr_3m_prices.values)
    assert np.abs(err).sum() < 3
    assert np.abs(err3).sum() < 3

def debug_sofr_swap_calibration():
    from curve import USDCurve
    sofr = USDCurve("SOFR", "2024-10-09")
    sofr.calibrate_swap_curve(sofr_swaps_rates)
    err = 1e2 * (sofr.price_spot_rates(sofr_swaps_rates.index) - sofr_swaps_rates.values)
    assert np.abs(err).sum() < 2

def debug_sofr_future_swap_convexity():
    from curve import USDCurve
    # SOFR1M and 3M
    sofr = USDCurve("SOFR", "2024-10-09")
    sofr.calibrate_future_curve(sofr_1m_prices, sofr_3m_prices)
    sofr.calibrate_swap_curve(sofr_swaps_rates)
    sofr.calculate_future_swap_spread()
    convexity = sofr.future_swap_spread
    assert np.round(convexity.abs().sum(), 2) == 0.57

def debug_swap_calibration_with_convexity():
    from curve import USDCurve
    sofr = USDCurve("SOFR", "2024-10-09")
    sofr.calibrate_future_curve(futures_3m_prices=sofr_3m_prices)
    convexity = pd.Series(1e-2 * np.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 11]),
                          index=range(16))
    sofr.calibrate_swap_curve(sofr_swaps_long, convexity)
    err = 1e2 * (sofr.price_spot_rates(sofr_swaps_long.index) - sofr_swaps_long.values)
    assert np.abs(err).max() < 0.2

def debug_shock_swap():
    from curve import USDCurve, shock_curve
    sofr = USDCurve("SOFR", "2024-10-09")
    sofr.calibrate_future_curve(futures_1m_prices=sofr_1m_prices, futures_3m_prices=sofr_3m_prices)
    sofr.calibrate_swap_curve(sofr_swaps_rates)
    old_zero_rates = sofr._swap_knot_values
    sofr.calculate_future_swap_spread()
    old_convexity = sofr.future_swap_spread.values.squeeze()

    sofr = shock_curve(sofr, "effective_rate", 10, "additive_bps", True)
    new_convexity = sofr.future_swap_spread.values.squeeze()
    new_zero_rates = sofr._swap_knot_values
    err = 1e2 * (old_convexity - new_convexity)
    assert np.abs(err).max() < 0.15

    bump = 1e4 * (new_zero_rates - old_zero_rates)
    assert np.round(np.abs(bump).mean(), 1) == 10.0

def debug_shock_future():
    from curve import USDCurve, shock_curve
    sofr = USDCurve("SOFR", "2024-10-09")
    sofr.calibrate_future_curve(futures_1m_prices=sofr_1m_prices, futures_3m_prices=sofr_3m_prices)
    sofr.calibrate_swap_curve(sofr_swaps_rates)
    old_effr = sofr.get_effective_rates()
    sofr.calculate_future_swap_spread()
    old_convexity = sofr.future_swap_spread.values.squeeze()

    sofr = shock_curve(sofr, "zero_rate", 10, "additive_bps", True)
    new_convexity = sofr.future_swap_spread.values.squeeze()
    new_effr = sofr.get_effective_rates()
    err = 1e2 * (old_convexity - new_convexity)
    assert np.abs(err).max() < 0.33

    bump = 1e2 * (new_effr - old_effr)
    assert np.round(np.abs(bump).mean(), 1) == 10.0

########################################################################################################################
# Vol testing
########################################################################################################################
sofr3m = pd.Series({
        "SFRU4": 95.2175,
        "SFRZ4": 95.635,
        "SFRH5": 96.05,
        "SFRM5": 96.355,
        "SFRU5": 96.53,
        "SFRZ5": 96.625,
        "SFRH6": 96.67,
        "SFRM6": 96.685,
        "SFRU6": 96.675,
    })
market_data = pd.Series({
    # Calls
    "SFRZ4C 95.25":	    0.3875,
    "SFRZ4C 95.375":	0.27,
    "SFRZ4C 95.50":	    0.1725,
    "SFRZ4C 95.625":	0.095,
    "SFRZ4C 95.75":	    0.0475,
    "SFRZ4C 95.875":	0.03,
    "SFRZ4C 96.00":	    0.02,
    "SFRZ4C 96.125":	0.015,
    "SFRZ4C 96.25":	    0.01,
    "SFRZ4C 96.375":	0.01,
    "SFRZ4C 96.50":	    0.0075,
    "SFRZ4C 96.625":    0.005,
    "SFRZ4C 96.75":     0.005,
    # Puts
    "SFRZ4P 95.875":	0.2675,
    "SFRZ4P 95.750":	0.1625,
    "SFRZ4P 95.625":	0.085,
    "SFRZ4P 95.500":	0.0375,
    "SFRZ4P 95.375":	0.0125,
    "SFRZ4P 95.250":	0.005,
    "SFRZ4P 95.125":	0.0025,
    "SFRZ4P 95.000":	0.0025,
    "SFRZ4P 94.875":    0.0025
})

def debug_option_parsing():
    from future_option import parse_sofr_option_ticker
    # Example usage:
    chain, exp, tick, exp_dt = parse_sofr_option_ticker('SFRH25')
    assert chain == 'SFRH5'
    assert exp == 'H5'
    assert tick == "SFRH5"
    assert exp_dt == dt.datetime(2025, 3, 14, 15)

    chain, exp, tick, exp_dt = parse_sofr_option_ticker('0QZ23')
    assert chain == '0QZ3'
    assert exp == 'Z3'
    assert tick == 'SFRZ4'
    assert exp_dt == dt.datetime(2023, 12, 15, 15, 0)


def debug_option_generation():
    from future_option import get_live_sofr_options, get_live_expiries
    # Generate tickers as of today
    today = dt.datetime.now()
    live = get_live_sofr_options(today)
    assert live[0] == "SFRX4"
    assert live[-1] == "SFRU8"
    assert len(live) == 74

    exp = get_live_expiries(today, "SFRZ4")
    assert exp == ['SFRX4', 'SFRZ4']


def debug_option_pricing():
    from curve import USDCurve
    from future_option import SOFRFutureOptionVolGrid
    ref_date = dt.datetime(2024, 10, 18)
    sofr = USDCurve("SOFR", ref_date).calibrate_future_curve(futures_3m_prices=sofr3m)
    vol_grid = SOFRFutureOptionVolGrid(sofr).load_option_data(market_data).calibrate_vol_grid()
    underlying, atm = vol_grid.atm("SFRZ4")
    assert underlying == 95.635
    assert np.round(atm, 4) == 0.5770


def debug_vol_gird_curve_change():
    from curve import USDCurve, shock_curve
    from future_option import SOFRFutureOptionVolGrid, shock_surface_curve
    ref_date = dt.datetime(2024, 10, 18)
    sofr = USDCurve("SOFR", ref_date).calibrate_future_curve(futures_3m_prices=sofr3m)
    sofr2 = shock_curve(sofr,
                        "effective_rate",
                        10,
                        "additive_bps",
                        False,
                        new_curve=True)
    vol_grid = SOFRFutureOptionVolGrid(sofr).load_option_data(market_data).calibrate_vol_grid()
    df1 = vol_grid.option_data["SFRZ4"]
    vol_grid2 = shock_surface_curve(vol_grid, sofr2, 0, True)
    df2 = vol_grid2.option_data["SFRZ4"]
    vol_chg = df2["vol"] - df1["vol"]
    assert np.round(vol_chg.mean(), 4) == 0.0295

def debug_vol_gird_shock():
    from curve import USDCurve, shock_curve
    from future_option import SOFRFutureOptionVolGrid, shock_surface_vol
    ref_date = dt.datetime(2024, 10, 18)
    sofr = USDCurve("SOFR", ref_date).calibrate_future_curve(futures_3m_prices=sofr3m)
    vol_grid = SOFRFutureOptionVolGrid(sofr).load_option_data(market_data).calibrate_vol_grid()
    df1 = vol_grid.option_data["SFRZ4"]
    vol_grid2 = shock_surface_vol(vol_grid, "SFRZ4", "additive", 0.1, True)
    df2 = vol_grid2.option_data["SFRZ4"]
    vol_chg = df2["vol"] - df1["vol"]
    assert np.all(np.round(vol_chg, 4) == 0.1)

def test_runner():
    total_tests = 0
    passed_tests = 0
    failed_tests = 0

    # Get the current module
    current_module = sys.modules[__name__]

    # Iterate over all members of the module
    for name, obj in inspect.getmembers(current_module):
        # Check if the member is a function and is not the test_runner itself
        if inspect.isfunction(obj) and obj.__module__ == current_module.__name__ and name != 'test_runner':
            total_tests += 1
            try:
                st = time.perf_counter()
                obj()
                passed_tests += 1
                elapsed_time = time.perf_counter() - st
                logging.info(f"{name}: Passed in {elapsed_time:.3f} seconds")
            except AssertionError as e:
                failed_tests += 1
                logging.info(f"{name}: Failed - {e}")
            except Exception as e:
                failed_tests += 1
                logging.info(f"{name}: Error - {e}")

    logging.info(f"Total tests: {total_tests}, Passed: {passed_tests}, Failed: {failed_tests}")

if __name__ == '__main__':
    test_runner()
