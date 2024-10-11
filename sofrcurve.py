import datetime as dt
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
import logging
logging.basicConfig(level=logging.INFO)
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import minimize, Bounds

from utils import convert_dates, parse_dates
from fixings import _SOFR_
from holiday import _SIFMA_
from swaps import SOFRSwap
from futures import SOFR1MFuture, SOFR3MFuture
from fomc import generate_fomc_meeting_dates


# USE SOFR curve class
class USDSOFRCurve:
    def __init__(self, reference_date):
        self.reference_date = pd.Timestamp(reference_date).to_pydatetime()
        self.future_knot_dates = None
        self.future_knot_values = None
        self.swap_knot_dates = None
        self.swap_knot_values = None
        self.convexity = None

    def initialize_future_knots(self, last_meeting_date: dt.datetime = None):
        """
        This function initializes the future curve knot dates given by FOMC meeting effective dates
        :return:
        """
        # Initialize future knots
        if last_meeting_date is None:
            last_meeting_date = self.reference_date + relativedelta(years=2)
        meeting_dates = generate_fomc_meeting_dates(self.reference_date, last_meeting_date)
        effective_dates = [_SIFMA_.next_biz_day(x, 1) for x in meeting_dates]
        next_biz_day = _SIFMA_.next_biz_day(self.reference_date, 0)

        if next_biz_day not in effective_dates:
            knot_dates = np.array([next_biz_day] + effective_dates)
        else:
            knot_dates = np.array(effective_dates)

        self.future_knot_dates = convert_dates(knot_dates)
        self.future_knot_values = 0.05 * np.ones((len(knot_dates), ))

    def initialize_swap_knots(self, swaps):
        """
        This function initializes the swap curve zero rate knots given by calibrating swap instruments
        :param swaps:
        :return:
        """
        # Initialize knots for swaps
        next_biz_day = _SIFMA_.next_biz_day(self.reference_date, 0)
        knot_dates = [SOFRSwap(self.reference_date, tenor=x).maturity_date for x in swaps]
        knot_dates = np.array([next_biz_day] + knot_dates)
        self.swap_knot_dates = convert_dates(knot_dates)
        self.swap_knot_values = 0.05 * np.ones((len(knot_dates), ))

    def plot_futures_daily_forwards(self, n_cuts=6):
        """
        Plots the future daily forward rates with additional annotations for the first n cuts.
        """
        # Assuming self.future_knot_values and self.future_knot_dates are defined elsewhere
        ind = parse_dates(self.future_knot_dates)
        val = 1e2 * self.future_knot_values
        df = pd.DataFrame(val, index=ind)

        # Create a dotted black line plot with step interpolation (left-continuous)
        plt.step(df.index, df.iloc[:, 0], where='post', linestyle=':', color='black')
        plt.scatter(df.index, df.iloc[:, 0], color='black', s=10, zorder=5)
        ax = plt.gca()
        ax.set_ylim(3.0, 5.25)
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.25))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, pos: '{:.2f}'.format(v)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%y' %b"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=90, ha='center')
        ax.grid(which='major', axis='y', linestyle='-', linewidth='0.5', color='gray', alpha=0.3)
        plt.xlabel('Date')
        plt.ylabel('SOFR Daily Forwards')
        plt.title('Constant Meeting-to-Meeting SOFR Daily Forwards Curve')

        # Adding annotations for the first six step-differences
        step_diffs = 1e2 * np.diff(df.iloc[:n_cuts+1, 0])  # Calculate the differences for the first 4 steps
        for i in range(n_cuts):
            x_pos = df.index[i + 1]
            y_pos = df.iloc[i + 1, 0]
            plt.annotate(f"{step_diffs[i]:.1f} bps", xy=(x_pos, y_pos),
                         xytext=(x_pos, y_pos + 0.05),  # Offset annotation slightly for clarity
                         fontsize=9, color='blue')

        plt.tight_layout()
        plt.show()
        return self

    def discount_factor(self, dates: np.array) -> np.array:
        return df(dates, self.swap_knot_dates, self.swap_knot_values, convert_dates(self.reference_date))

    def plot_swap_zero_rate(self):
        ind = parse_dates(self.swap_knot_dates)
        val = 1e2 * self.swap_knot_values
        df = pd.DataFrame(val, index=ind)
        plt.plot(df.index, df.values)
        plt.xlabel('Date')
        plt.ylabel('Zero Rates')
        plt.title('Continuously Compounded Zero Coupon Rate Curve')
        plt.tight_layout()
        plt.show()
        return self

    def calibrate_convexity(self):
        pass


def df(dates: np.array, knot_dates: np.array, knot_values: np.array, ref_date: float):
    """
    Hard to jit because of the interpolation.
    :param dates:
    :param knot_dates:
    :param knot_values:
    :param ref_date:
    :return:
    """
    zero_rates = np.interp(dates, knot_dates, knot_values)
    t_vect = (dates - ref_date) / 360
    return np.exp(-zero_rates * t_vect)


def par_rate(ref_date: float, schedule: np.array, knot_dates: np.array, knot_values: np.array):
    """
    This function prices a single swap's par rate based on its schedule.
    :param ref_date:
    :param schedule:
    :param knot_dates:
    :param knot_values:
    :return:
    """
    sd = schedule[:, 0]
    ed = schedule[:, 1]
    pay = schedule[:, 2]
    dcf = schedule[:, 3]
    df_s = df(sd, knot_dates, knot_values, ref_date)
    df_e  = df(ed, knot_dates, knot_values, ref_date)
    df_p = df(pay, knot_dates, knot_values, ref_date)
    fwd_rate = df_s / df_e - 1
    numerator = np.sum(df_p * fwd_rate)
    denominator = np.sum(dcf * df_p)
    return numerator / denominator

def price_swap_rates(curve: USDSOFRCurve, swaps: list) -> np.array:
    """
        This function prices a list of spot starting swaps on a curve.
        :param curve:
        :param swaps:
        :return:
        """
    st = time.perf_counter()
    ref_date = convert_dates(curve.reference_date)
    schedules = [swap.get_float_leg_schedule(True).values for swap in swaps]
    knot_dates = curve.swap_knot_dates
    knot_values = curve.swap_knot_values
    rates = np.array(list(map(lambda schedule: par_rate(ref_date, schedule, knot_dates, knot_values), schedules)))
    logging.info(f"Priced {len(swaps)} spot swaps in {time.perf_counter() - st:.3f}s")
    return 1e2 * rates

def spot_swap_rates(curve: USDSOFRCurve, swaps_list) -> np.array:
    swaps = [SOFRSwap(curve.reference_date, tenor=tenor) for tenor in swaps_list]
    return price_swap_rates(curve, swaps)

def calibrate_swap_curve(curve: USDSOFRCurve, swaps: pd.Series):
    """
    This function calibrates a swap curve's zero rate knots to prices swaps according to an input market
    :param curve:
    :param swaps:
    :return:
    """
    st = time.perf_counter()
    curve.initialize_swap_knots(swaps.index)
    ref_date = convert_dates(curve.reference_date)
    schedules = [SOFRSwap(curve.reference_date, tenor=x).get_float_leg_schedule(True).values for x in swaps.index]
    mkt_prices = swaps.values.squeeze()
    knot_dates = curve.swap_knot_dates
    initial_values = 0.05 * np.ones_like(curve.swap_knot_values)

    def loss_function(knot_values: np.array) -> float:
        rates = 1e2 * np.array(list(map(lambda schedule: par_rate(ref_date, schedule, knot_dates, knot_values), schedules)))
        loss = np.sum((rates - mkt_prices) ** 2)
        loss += 1e2 * np.sum(np.diff(knot_values) ** 2)
        return loss

    bounds = Bounds(0.0, 0.08)
    res = minimize(loss_function,
                   initial_values,
                   method="L-BFGS-B",
                   bounds=bounds)
    curve.swap_knot_values = res.x
    logging.info(f"Calibrated swap curve in {time.perf_counter() - st:.3f}s")

def create_overlap_matrix(start_end_dates: np.ndarray, knot_dates: np.ndarray) -> np.ndarray:
    """
    This function creates overlap matrix, how many calendar days is each future exposed to each meeting-to-meeting period
    :param start_end_dates:
    :param knot_dates:
    :return:
    """
    knot_ends = np.roll(knot_dates, -1) - 1
    knot_ends[-1] = 1_000_000
    starts = np.maximum(start_end_dates[:, 0].reshape(-1, 1), knot_dates.reshape(1, -1))
    ends = np.minimum(start_end_dates[:, 1].reshape(-1, 1), knot_ends.reshape(1, -1))
    return np.maximum(0, ends - starts + 1)


def price_1m_futures(curve: USDSOFRCurve, futures_1m: list) -> np.ndarray:
    """
    This function values a list of futures_1m (tickers) on a given curve
    :param curve:
    :param futures_1m:
    :return:
    """
    st = time.perf_counter()
    fut_start_end = np.array([convert_dates(SOFR1MFuture(x).get_reference_start_end_dates()) for x in futures_1m])

    front_fixing = 0
    front_future = SOFR1MFuture(futures_1m[0])
    if front_future.reference_start_date < curve.reference_date:
        fixings = 1e-2 * _SOFR_.get_fixings_asof(front_future.reference_start_date, curve.reference_date - dt.timedelta(days=1))
        front_fixing = fixings.sum()

    o_matrix = create_overlap_matrix(fut_start_end, curve.future_knot_dates)
    knot_values = curve.future_knot_values

    rate_sum = np.matmul(o_matrix, knot_values.reshape(-1, 1)).squeeze()
    rate_sum[0] += front_fixing

    rate_avg = rate_sum / (np.diff(fut_start_end, axis=1) + 1).squeeze()
    logging.info(f"Priced {len(futures_1m)} 1m futures in {time.perf_counter()-st:.3f}s")
    return 1e2 * (1 - rate_avg)


def price_3m_futures(curve: USDSOFRCurve, futures_3m: list) -> np.ndarray:
    """
    This function values a list of futures_3m (tickers) on a given curve
    :param curve:
    :param futures_3m:
    :return:
    """
    st = time.perf_counter()
    fut_start_end = np.array([convert_dates(SOFR3MFuture(x).get_reference_start_end_dates()) for x in futures_3m])

    front_fixing = 1
    front_future = SOFR3MFuture(futures_3m[0])
    if front_future.reference_start_date < curve.reference_date:
        fixings = 1e-2 * _SOFR_.get_fixings_asof(front_future.reference_start_date, curve.reference_date - dt.timedelta(days=1))
        front_fixing = (1 + fixings / 360).prod()

    o_matrix = create_overlap_matrix(fut_start_end, curve.future_knot_dates)
    knot_values = curve.future_knot_values

    rate_prod = np.exp(np.matmul(o_matrix, np.log(1 + knot_values.reshape(-1, 1) / 360)).squeeze())
    rate_prod[0] *= front_fixing

    rate_avg = 360 * (rate_prod - 1) / (np.diff(fut_start_end, axis=1) + 1).squeeze()
    logging.info(f"Priced {len(futures_3m)} 1m futures in {time.perf_counter()-st:.3f}s")
    return 1e2 * (1 - rate_avg)


def objective_function(knot_values: np.ndarray,
                       o_mat_1m: np.ndarray, fixing_1m: float, n_days_1m: np.ndarray, prices_1m: np.ndarray,
                       o_mat_3m: np.ndarray, fixing_3m: float, n_days_3m: np.ndarray, prices_3m: np.ndarray):
    # Price the 1m futures
    rate_sum = o_mat_1m @ knot_values.transpose()
    rate_sum[0] += fixing_1m
    rate_avg_1m = rate_sum / n_days_1m
    px_1m = 1e2 * (1 - rate_avg_1m)
    loss = np.sum((px_1m - prices_1m) ** 2)

    # Price the 3m futures
    rate_prod = np.exp(o_mat_3m @ np.log(1 + knot_values.transpose() / 360))
    rate_prod[0] *= fixing_3m
    rate_avg_3m = 360 * (rate_prod - 1) / n_days_3m
    px_3m = 1e2 * (1 - rate_avg_3m)
    loss += np.sum((px_3m - prices_3m) ** 2)

    # Add a little curve constraint loss
    loss += 1e2 * np.sum(np.diff(knot_values) ** 2)
    return loss

def calibrate_futures_curve(curve: USDSOFRCurve, futures_1m: pd.Series, futures_3m: pd.Series):
    st = time.perf_counter()

    # Initialize the FOMC meeting effective dates up till expiry of the last 3m future
    last_future = SOFR3MFuture(futures_3m.index[-1])
    curve.initialize_future_knots(last_future.reference_end_date)

    # Obtain the start and end dates of the 1m and 3m futures
    fut_start_end_1m = np.array([convert_dates(SOFR1MFuture(x).get_reference_start_end_dates()) for x in futures_1m.index])
    n_days_1m = (np.diff(fut_start_end_1m, axis=1) + 1).squeeze()
    fut_start_end_3m = np.array([convert_dates(SOFR3MFuture(x).get_reference_start_end_dates()) for x in futures_3m.index])
    n_days_3m = (np.diff(fut_start_end_3m, axis=1) + 1).squeeze()

    # Take the front 1m future stub sum as well as the front 3m future stub prod
    front_fixing_sum = 0.0
    front_future_1m = SOFR1MFuture(futures_1m.index[0])
    if front_future_1m.reference_start_date < curve.reference_date:
        fixings = 1e-2 * _SOFR_.get_fixings_asof(front_future_1m.reference_start_date,
                                                 curve.reference_date - dt.timedelta(days=1))
        front_fixing_sum = fixings.sum()

    front_fixing_prod = 1.0
    front_future_3m = SOFR3MFuture(futures_3m.index[0])
    if front_future_3m.reference_start_date < curve.reference_date:
        fixings = 1e-2 * _SOFR_.get_fixings_asof(front_future_3m.reference_start_date,
                                                 curve.reference_date - dt.timedelta(days=1))
        front_fixing_prod = (1 + fixings / 360).prod()

    # Create the overlap matrices for 1m and 3m futures for knot periods
    o_matrix_1m = create_overlap_matrix(fut_start_end_1m, curve.future_knot_dates).astype(float)
    o_matrix_3m = create_overlap_matrix(fut_start_end_3m, curve.future_knot_dates).astype(float)

    # Initial values
    initial_knot_values = curve.future_knot_values
    bounds = Bounds(0.0, 0.08)
    res = minimize(objective_function,
                   initial_knot_values,
                   args=(o_matrix_1m, front_fixing_sum, n_days_1m, futures_1m.values.squeeze(),
                         o_matrix_3m, front_fixing_prod, n_days_3m, futures_3m.values.squeeze()),
                   method="L-BFGS-B",
                   bounds=bounds)
    curve.future_knot_values = res.x

    logging.info(f"Finished futures curve calibration in {time.perf_counter()-st:.3f}s")

def compute_convexity(curve: USDSOFRCurve, futures_3m):
    """
    This function uses curve to evaluate future prices as well as equivalent swap rates.
    :param curve:
    :param futures_3m:
    :return:
    """
    future_rates = 1e2 - price_3m_futures(curve, futures_3m)
    fut_start_end_dates = [SOFR3MFuture(x).get_reference_start_end_dates() for x in futures_3m]
    for st_et in fut_start_end_dates:
        st_et[1] += dt.timedelta(1)
    swaps = [SOFRSwap(start_date=x, maturity_date=y) for x, y in fut_start_end_dates]
    swap_rates = price_swap_rates(curve, swaps)
    return swap_rates - future_rates

# Example usage
if __name__ == '__main__':
    sofr_1m_prices = pd.Series({
        "SERV24": 95.155,
        "SERX24": 95.310,
        "SERZ24": 95.435,
        "SERF25": 95.605,
        "SERG25": 95.795,
        "SERH25": 95.870,
        "SERJ25": 96.000,
        "SERK25": 96.105,
        "SERM25": 96.185,
        "SERN25": 96.275,
        "SERQ25": 96.355,
        "SERU25": 96.435,
    }, name="SOFR1M")
    sofr_3m_prices = pd.Series({
        "SFRU24": 95.205,
        "SFRZ24": 95.680,
        "SFRH25": 96.045,
        "SFRM25": 96.300,
        "SFRU25": 96.465,
        "SFRZ25": 96.560,
        "SFRH26": 96.610,
        "SFRM26": 96.625,
        "SFRU26": 96.620,
        "SFRZ26": 96.610,
        "SFRH27": 96.605,
        "SFRM27": 96.600,
        "SFRU27": 96.590,
        "SFRZ27": 96.575,
        "SFRH28": 96.560,
        "SFRM28": 96.545,
        "SFRU28": 96.525,
    }, name="SOFR3M")
    sofr_swaps_rates = pd.Series({
        "1W": 4.8400,
        "2W": 4.8318,
        "3W": 4.8455,
        "1M": 4.8249,
        "2M": 4.7530,
        "3M": 4.6709,
        "6M": 4.4717,
        "9M": 4.3061,
        "1Y": 4.16675,
        "18M": 3.9378,
        "2Y": 3.81955,
        "3Y": 3.6866,
        "4Y": 3.61725,
        "5Y": 3.5842,
        "7Y": 3.5719,
        "10Y": 3.5972,
        "15Y": 3.6590,
        "20Y": 3.6614,
        "30Y": 3.51965
    })

    sofr = USDSOFRCurve("2024-10-09")
    calibrate_futures_curve(sofr, sofr_1m_prices, sofr_3m_prices)
    calibrate_swap_curve(sofr, sofr_swaps_rates)

    fut_1m = price_1m_futures(sofr, sofr_1m_prices.index)
    print("Pricing errors in bps for SOFR1M futures:")
    print(1e2 * (fut_1m - sofr_1m_prices.values))

    fut_3m = price_3m_futures(sofr, sofr_3m_prices.index)
    print("Pricing errors in bps for SOFR3M futures:")
    print(1e2 * (fut_3m - sofr_3m_prices.values))

    swap_rates = spot_swap_rates(sofr, sofr_swaps_rates.index)
    print("Pricing errors in bps for swaps")
    print(1e2 * (swap_rates - sofr_swaps_rates.values))

    convexity = compute_convexity(sofr, futures_3m=sofr_3m_prices.index[1:])
    print("Future-Swap convexity in bps:")
    print(1e2 * convexity)

    sofr.plot_futures_daily_forwards(6)
    sofr.plot_swap_zero_rate()
    exit(0)
