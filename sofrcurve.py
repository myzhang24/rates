import datetime as dt
import numpy as np
import pandas as pd
from jaxopt import ScipyBoundedMinimize
from holiday import SIFMA
from swaps import SOFRSwap
from futures import SOFR1MFutures, SOFR3MFutures
from fomc import generate_fomc_meeting_dates
from dateutil.relativedelta import relativedelta
from fixings import load_SOFR_fixings
from jax import jit
import jax.numpy as jnp


# Use 1904 date format
base_date_1904 = dt.datetime(1904, 1, 1)

def convert_to_1904(dates: pd.DatetimeIndex | np.ndarray | list) -> jnp.array:
    # If it's a pandas DatetimeIndex, convert to an array of datetime
    if isinstance(dates, pd.DatetimeIndex):
        dates = dates.to_pydatetime()

    # Convert each date to Excel 1904 integer format
    return jnp.array([(dt - base_date_1904).days for dt in dates])

# Some Jax pricing routines
@jit
def last_published_value(reference_dates: jnp.ndarray,
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
def sofr_compound(reference_dates: jnp.ndarray,
                  reference_rates: jnp.ndarray):
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

        self.market_instruments = {}
        self.sofr_1m_futures = []
        self.sofr_3m_futures = []
        self.sofr_fras = []
        self.sofr_swaps = []

        self.future_curve_tenor = 2
        self.future_knot_dates = None
        self.future_knot_values = None
        self.initialize_future_knot_dates()

        self.swap_knot_policy = "Payment Date"
        self.swap_knot_dates = None
        self.swap_knot_values = None

        self.fixings = 1e-2 * load_SOFR_fixings(self.reference_date - relativedelta(months=4), self.reference_date)

    def initialize_future_knot_dates(self):
        """
        This function initializes the future curve knot dates given by FOMC meeting effective dates
        :return:
        """
        # Initialize future knots
        far_out_date = self.reference_date + pd.DateOffset(years=self.future_curve_tenor)
        meeting_dates = generate_fomc_meeting_dates(self.reference_date, far_out_date)
        effective_dates = [SIFMA.next_biz_day(x, 1) for x in meeting_dates]
        next_biz_day = SIFMA.next_biz_day(self.reference_date, 0)
        if next_biz_day not in effective_dates:
            self.future_knot_dates = np.array([next_biz_day] + effective_dates)
        else:
            self.future_knot_dates = np.array(effective_dates)

    def initialize_swap_knot_dates(self):
        # Initialize knots for swaps
        next_biz_day = SIFMA.next_biz_day(self.reference_date, 0)
        swap_dates = [x.fixed_leg_schedule.iloc[-1].loc[self.swap_knot_policy] for x in self.sofr_swaps]
        self.swap_knot_dates = np.array([next_biz_day] + swap_dates)

    def load_market_data(self, sofr_3m_futures, sofr_1m_futures=None, sofr_swaps=None, sofr_fras=None):
        """
        Load market data for calibration.

        Parameters:
        - sofr_1m_futures: 2d array of [ticker, price]
        - sofr_3m_futures: 2d array of [ticker, price]
        - sofr_fras: 2d array of [shorthand, rate]
        - sofr_ois_swaps: 2d array of [shorthand, rate]
        """

        self.market_instruments = {
            "SOFR3M": sofr_3m_futures,
            "SOFR1M": sofr_1m_futures,
            "SOFRFRAs": sofr_fras,
            "SOFRSwaps": sofr_swaps
        }

        # Load market price into 1m futures instances
        for key, price in sofr_3m_futures.items():
            self.sofr_3m_futures.append(SOFR3MFutures(key.upper()))

        if sofr_1m_prices is not None:
            for key, price in sofr_1m_futures.items():
                self.sofr_1m_futures.append(SOFR1MFutures(key.upper()))

        if sofr_swaps is not None:
            for key, rate in sofr_swaps.items():
                start_date = SIFMA.next_biz_day(self.reference_date, 2)
                self.sofr_swaps.append(SOFRSwap(start_date, key.upper()))

        self.initialize_swap_knot_dates()
        return self

    def swap_reference_arrays(self):
        pass

    def future_1m_reference_arrays(self):
        # 1M futures
        ref_dates_1m = []
        for fut in self.sofr_1m_futures:
            ref_dates_1m.append(convert_to_1904(fut.reference_array()))
        return ref_dates_1m

    def future_3m_reference_arrays(self):
        # 3M futures
        ref_dates_3m = []
        for fut in self.sofr_3m_futures:
            ref_dates_3m.append(convert_to_1904(fut.reference_array()))
        return ref_dates_3m


    def build_future_curve(self):
        # Prepare the local constants
        ref_array_1m = self.future_1m_reference_arrays()
        ref_array_3m = self.future_3m_reference_arrays()

        fut_price_1m = jnp.array(self.market_instruments["SOFR1M"].values)
        fut_price_3m = jnp.array(self.market_instruments["SOFR3M"].values)

        fixing_dates = convert_to_1904(self.fixings.index)
        fixing_values = jnp.array(self.fixings.values)

        knot_dates = convert_to_1904(self.future_knot_dates)
        initial_values = 0.05 * jnp.ones_like(knot_dates, dtype=float)

        penalty_1m = 0.25 * jnp.ones_like(fut_price_1m)
        penalty_1m.at[1:4].set(0.75)

        penalty_3m = jnp.ones_like(fut_price_3m)
        penalty_3m.at[0].set(0.25)
        penalty_3m.at[7:].set(0.5)

        # Write the objective function
        def futures_objective_function(knot_values):
            """
            Build the constant meeting daily forward futures curve
            :return:
            """
            prices_1m = jnp.zeros_like(fut_price_1m)
            prices_3m = jnp.zeros_like(fut_price_3m)

            knot_to_use = jnp.concatenate([fixing_dates, knot_dates])
            value_to_use = jnp.concatenate([fixing_values, knot_values])

            for i in range(prices_1m.shape[0]):
                price = 1e2 * ( 1 - jnp.mean(last_published_value(ref_array_1m[i], knot_to_use, value_to_use)).squeeze())
                prices_1m = prices_1m.at[i].set(price)

            for i in range(prices_3m.shape[0]):
                rates = last_published_value(ref_array_3m[i], knot_to_use, value_to_use)
                price = 1e2 * (1 - sofr_compound(ref_array_3m[i], rates).squeeze())
                prices_3m = prices_3m.at[i].set(price)

            res = jnp.sum(penalty_1m * (prices_1m - fut_price_1m) ** 2)
            res += jnp.sum(penalty_3m * (prices_3m - fut_price_3m) ** 2)
            res += 1e2 * jnp.sum(jnp.diff(knot_values) ** 2)
            return res

        # Use jax lbfgsb to minimize with jit and autodiff
        lbfgsb = ScipyBoundedMinimize(fun=futures_objective_function, method="l-bfgs-b", jit=True)
        lower_bounds = jnp.zeros_like(initial_values)
        upper_bounds = jnp.ones_like(initial_values) * 0.1
        bounds = (lower_bounds, upper_bounds)
        res = lbfgsb.run(initial_values, bounds=bounds).params
        self.future_knot_values = res
        return self

    def price_1m_futures(self):
        fut_price_1m = jnp.array(self.market_instruments["SOFR1M"].values)
        prices_1m = jnp.zeros_like(fut_price_1m)
        ref_array_1m = self.future_1m_reference_arrays()
        fixing_dates = convert_to_1904(self.fixings.index)
        fixing_values = jnp.array(self.fixings.values)
        knot_dates = convert_to_1904(self.future_knot_dates)
        knot_values = jnp.array(self.future_knot_values)

        knot_to_use = jnp.concatenate([fixing_dates, knot_dates])
        value_to_use = jnp.concatenate([fixing_values, knot_values])

        for i in range(prices_1m.shape[0]):
            price = 1e2 * (1 - jnp.mean(last_published_value(ref_array_1m[i], knot_to_use, value_to_use)).squeeze())
            prices_1m = prices_1m.at[i].set(price)
        return prices_1m

    def price_3m_futures(self):
        fut_price_3m = jnp.array(self.market_instruments["SOFR3M"].values)
        prices_3m = jnp.zeros_like(fut_price_3m)
        ref_array_3m = self.future_3m_reference_arrays()
        fixing_dates = convert_to_1904(self.fixings.index)
        fixing_values = jnp.array(self.fixings.values)
        knot_dates = convert_to_1904(self.future_knot_dates)
        knot_values = jnp.array(self.future_knot_values)

        knot_to_use = jnp.concatenate([fixing_dates, knot_dates])
        value_to_use = jnp.concatenate([fixing_values, knot_values])

        for i in range(prices_3m.shape[0]):
            rates = last_published_value(ref_array_3m[i], knot_to_use, value_to_use)
            price = 1e2 * (1 - sofr_compound(ref_array_3m[i], rates).squeeze())
            prices_3m = prices_3m.at[i].set(price)
        return prices_3m

    def plot_future_daily_forwards(self):
        """
        Plots the future daily forward rates with additional annotations for the first four step differences.
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        # Assuming self.future_knot_values and self.future_knot_dates are defined elsewhere
        df = 1e2 * pd.DataFrame(np.array(self.future_knot_values), index=self.future_knot_dates)

        # Create a dotted black line plot with step interpolation (left-continuous)
        plt.step(df.index, df.iloc[:, 0], where='post', linestyle=':', color='black')
        plt.scatter(df.index, df.iloc[:, 0], color='black', s=10, zorder=5)
        ax = plt.gca()
        ax.set_ylim(3.0, 5.25)
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.25))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: '{:.2f}'.format(val)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%y' %b"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=90, ha='center')
        ax.grid(which='major', axis='y', linestyle='-', linewidth='0.5', color='gray', alpha=0.3)
        plt.xlabel('Date')
        plt.ylabel('SOFR Daily Forwards')
        plt.title('Constant Meeting-to-Meeting SOFR Daily Forwards Curve')

        # Adding annotations for the first six step-differences
        step_diffs = 1e2 * np.diff(df.iloc[:7, 0])  # Calculate the differences for the first 4 steps
        for i in range(6):
            x_pos = df.index[i + 1]
            y_pos = df.iloc[i + 1, 0]
            plt.annotate(f"{step_diffs[i]:.1f} bps", xy=(x_pos, y_pos),
                         xytext=(x_pos, y_pos + 0.05),  # Offset annotation slightly for clarity
                         fontsize=9, color='blue')

        plt.tight_layout()
        plt.show()

        return self

# Example usage
if __name__ == '__main__':
    sofr_1m_prices = pd.Series({
        "SERV24": 95.15,
        "SERX24": 95.315,
        "SERZ24": 95.455,
        "SERF25": 95.64,
        "SERG25": 95.84,
        "SERH25": 95.925,
        "SERJ25": 96.06,
        "SERK25": 96.175,
        # "SERM25": 96.,
        "SERN25": 96.345,
        "SERQ25": 96.425,
        # "SERU25": 96.415,
    }, name="SOFR1M")
    sofr_3m_prices = pd.Series({
        "SFRU24": 95.2125,
        "SFRZ24": 95.71,
        "SFRH25": 96.105,
        "SFRM25": 96.365,
        "SFRU25": 96.53,
        "SFRZ25": 96.625,
        "SFRH26": 96.67,
        "SFRM26": 96.685,
        "SFRU26": 96.68
    }, name="SOFR3M")
    sofr_swaps_rates = pd.Series({
        "1Y": 0.0389,
        "3Y": 0.0338,
        "5Y": 0.0332,
        "7Y": 0.0333,
        "10Y": 0.0338,
        "15Y": 0.0347,
        "30Y": 0.0336
    })

    curve = USDSOFRCurve("2024-10-08")
    curve.load_market_data(sofr_3m_prices, sofr_1m_prices, sofr_swaps_rates)
    curve.build_future_curve()
    curve.plot_future_daily_forwards()

    exit(0)
