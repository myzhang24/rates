import datetime as dt
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
import logging
logging.basicConfig(level=logging.INFO)
import time

from jax import jit
import jax.numpy as jnp
from jaxopt import ScipyBoundedMinimize

from utils import convert_dates, parse_dates
from fixings import _SOFR_
from holiday import _SIFMA_
from swaps import SOFRSwap
from futures import SOFR1MFuture, SOFR3MFuture
from fomc import generate_fomc_meeting_dates




# Some Jax pricing routines
@jit
def last_known_value(reference_dates: jnp.ndarray,
                         knot_dates: jnp.ndarray,
                         knot_values: jnp.ndarray) -> jnp.ndarray:
    """
    This function looks up reference_values for reference_dates according to knot_dates, knot_values
    :param reference_dates:
    :param knot_dates:
    :param knot_values:
    :return:
    """
    indices = jnp.searchsorted(knot_dates, reference_dates, side='right') - 1
    indices = jnp.clip(indices, 0, len(knot_values) - 1)
    return knot_values[indices]

@jit
def ois_compound(reference_dates: jnp.ndarray, reference_rates: jnp.ndarray):
    """
    This function computes the compounded SOFR rate given the fixings
    :param reference_dates:
    :param reference_rates:
    :return:
    """
    num_days = jnp.diff(reference_dates)
    rates = reference_rates[:-1]
    annualized_rate = jnp.prod(1 + rates * num_days / 360) - 1
    return 360 * annualized_rate / num_days.sum()


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

    def initialize_swap_knots(self, swaps, policy="Termination Date"):
        """
        This function initializes the swap curve zero rate knots given by calibrating swap instruments
        :param swaps:
        :param policy:
        :return:
        """
        # Initialize knots for swaps
        next_biz_day = _SIFMA_.next_biz_day(self.reference_date, 0)
        knot_dates = [x.specs[policy] for x in swaps]
        knot_dates = np.array([next_biz_day] + knot_dates)
        self.swap_knot_dates = convert_dates(knot_dates)
        self.swap_knot_values = 0.05 * np.ones((len(knot_dates), ))

    def plot_futures_daily_forwards(self, n_cuts=6):
        """
        Plots the future daily forward rates with additional annotations for the first n cuts.
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

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

    def calibrate_convexity(self):
        pass


def price_1m_futures(curve: USDSOFRCurve, futures_1m: list) -> jnp.ndarray:
    """
    This function values a list of futures_1m (tickers) on a given curve
    :param curve:
    :param futures_1m:
    :return:
    """
    st = time.perf_counter()
    ref_periods = [jnp.array(SOFR1MFuture(x).reference_array()) for x in futures_1m]
    fixings = 1e-2 * _SOFR_.get_fixings(curve.reference_date - relativedelta(months=1), curve.reference_date)
    fixing_dates = convert_dates(fixings.index)
    knots = jnp.concatenate([fixing_dates, curve.future_knot_dates])
    values = jnp.concatenate([fixings.values.squeeze(), curve.future_knot_values])

    res = jnp.zeros((len(futures_1m), ))
    for i, ref_period in enumerate(ref_periods):
        ref_rates = last_known_value(ref_period, knots, values)
        rate = ref_rates.mean()
        res = res.at[i].set(jnp.round(1e2 * (1 - rate), 3))
    logging.info(f"Priced {len(futures_1m)} 1m futures in {time.perf_counter()-st:.3f}s")
    return res


def price_3m_futures(curve: USDSOFRCurve, futures_3m: list) -> jnp.ndarray:
    """
    This function values a list of futures_1m (tickers) on a given curve
    :param curve:
    :param futures_3m:
    :return:
    """
    st = time.perf_counter()
    ref_periods = [jnp.array(SOFR3MFuture(x).reference_array()) for x in futures_3m]
    fixings = 1e-2 * _SOFR_.get_fixings(curve.reference_date - relativedelta(months=4), curve.reference_date)
    fixing_dates = convert_dates(fixings.index)
    knots = jnp.concatenate([fixing_dates, curve.future_knot_dates])
    values = jnp.concatenate([fixings.values.squeeze(), curve.future_knot_values])

    res = jnp.zeros((len(futures_3m), ))
    for i, ref_period in enumerate(ref_periods):
        ref_rates = last_known_value(ref_period, knots, values)
        rate = ois_compound(ref_period, ref_rates)
        res = res.at[i].set(1e2 * (1 - rate))
    logging.info(f"Priced {len(futures_3m)} 1m futures in {time.perf_counter() - st:.3f}s")
    return res

def price_3m_futures_approx(curve: USDSOFRCurve, futures_3m: list) -> jnp.ndarray:
    ref_periods = [jnp.array(SOFR3MFuture(x).reference_array(biz_only=False)) for x in futures_3m]
    fixings = 1e-2 * _SOFR_.get_fixings(curve.reference_date - relativedelta(months=4), curve.reference_date)
    fixing_dates = convert_dates(fixings.index)
    knots = jnp.concatenate([fixing_dates, curve.future_knot_dates])
    values = jnp.concatenate([fixings.values.squeeze(), curve.future_knot_values])

    res = jnp.zeros((len(futures_3m),))
    for i, ref_period in enumerate(ref_periods):
        ref_rates = last_known_value(ref_period, knots, values)
        rate = 360 * ((1 + ref_rates / 360).prod() - 1) / (ref_period[-1] - ref_period[0] + 1)
        res = res.at[i].set(1e2 * (1 - rate))
    return res

def calibrate_futures_curve(curve: USDSOFRCurve, futures_1m: pd.Series, futures_3m: pd.Series):
    st = time.perf_counter()

    ref_periods_1m = [jnp.array(SOFR1MFuture(x).reference_array()) for x in futures_1m.index]
    ref_periods_3m = [jnp.array(SOFR3MFuture(x).reference_array()) for x in futures_3m.index]
    curve.initialize_future_knots(SOFR3MFuture(futures_3m.index[-1]).reference_end_date)

    fixings = 1e-2 * _SOFR_.get_fixings(curve.reference_date - relativedelta(months=4), curve.reference_date)
    fixing_dates = convert_dates(fixings.index)
    knots = jnp.concatenate([fixing_dates, curve.future_knot_dates])
    fixing_values = jnp.array(fixings.values.squeeze())

    px_1m = jnp.zeros((len(futures_1m), ))
    px_3m = jnp.zeros((len(futures_3m), ))
    market_1m = jnp.array(futures_1m.values.squeeze())
    market_3m = jnp.array(futures_3m.values.squeeze())

    initial_values = jnp.array(curve.future_knot_values)

    @jit
    def futures_objective_function(knot_values, prices_1m, prices_3m):
        """
        Build the constant meeting daily forward futures curve
        :return:
        """
        values = jnp.concatenate([fixing_values, knot_values])

        for i in range(prices_1m.shape[0]):
            ref_period = ref_periods_1m[i]
            ref_rates = last_known_value(ref_period, knots, values)
            prices_1m = prices_1m.at[i].set(1e2 * (1 - ref_rates.mean()))

        for i in range(prices_3m.shape[0]):
            ref_period = ref_periods_3m[i]
            ref_rates = last_known_value(ref_period, knots, values)
            rate = ois_compound(ref_period, ref_rates)
            prices_3m = prices_3m.at[i].set(1e2 * (1 - rate))

        score = jnp.sum((prices_1m - market_1m) ** 2)
        score += jnp.sum((prices_3m - market_3m) ** 2)
        score += 1e2 * jnp.sum(jnp.diff(knot_values) ** 2)
        return score

    # Use jax lbfgsb to minimize with jit and autodiff
    lbfgsb = ScipyBoundedMinimize(fun=futures_objective_function,
                                  method="l-bfgs-b", jit=True)
    lower_bounds = jnp.zeros_like(initial_values)
    upper_bounds = jnp.ones_like(initial_values) * 0.08
    bounds = (lower_bounds, upper_bounds)
    res = lbfgsb.run(initial_values,
                     prices_1m=px_1m,
                     prices_3m=px_3m,
                     bounds=bounds).params
    curve.future_knot_values = np.array(res)

    logging.info(f"Finished futures curve calibration in {time.perf_counter()-st:.3f}s")

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
        "SFRU26": 96.620
    }, name="SOFR3M")
    sofr_swaps_rates = pd.Series({
        "1W": 4.8400,
        "1M": 4.8249,
        "3M": 4.6709,
        "6M": 4.4717,
        "1Y": 4.16675,
        "2Y": 3.81955,
        "3Y": 3.6866,
        "5Y": 3.5842,
        "7Y": 3.5719,
        "10Y": 3.5972,
        "15Y": 3.6590,
        "30Y": 3.51965
    })

    sofr = USDSOFRCurve("2024-10-09")
    calibrate_futures_curve(sofr, sofr_1m_prices, sofr_3m_prices)
    # fut_1m = price_1m_futures(sofr, sofr_1m_prices.index)
    # print(1e2 * (fut_1m - sofr_1m_prices.values))
    fut_3m = price_3m_futures(sofr, sofr_3m_prices.index)
    fut_3m_approx = price_3m_futures_approx(sofr, sofr_3m_prices.index)
    print(1e2 * (fut_3m - fut_3m_approx))


    # sofr.plot_futures_daily_forwards(6)
    # print(sofr.future_knot_values)
    exit(0)
