import datetime as dt
import numpy as np
import pandas as pd
from jaxopt import ScipyBoundedMinimize
from holiday import SIFMA
from swaps import SOFRSwap, SOFRFRA, fra_start_end_date
from futures import SOFR1MFutures, SOFR3MFutures
from fomc import generate_fomc_meeting_dates
from dateutil.relativedelta import relativedelta
from fixings import load_fixings
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

        if sofr_fras is not None:
            for key, rate in sofr_fras.items():
                start_date, end_date = fra_start_end_date(self.reference_date, key.upper())
                self.sofr_fras.append(SOFRFRA(start_date, end_date))

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
        ref_array_1m = self.future_1m_reference_arrays()
        ref_array_3m = self.future_3m_reference_arrays()

        fut_price_1m = jnp.array(self.market_instruments["SOFR1M"].values)
        fut_price_3m = jnp.array(self.market_instruments["SOFR3M"].values)

        fixings = 1e-2 * load_fixings(self.reference_date - relativedelta(months=4), self.reference_date)
        fixing_dates = convert_to_1904(fixings.index)
        fixing_values = jnp.array(fixings.values)

        last_fixing = fixing_values[-1]

        knot_dates = convert_to_1904(self.future_knot_dates)
        initial_values = 0.05 * jnp.ones_like(knot_dates, dtype=float)

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

            res = 0.5 * jnp.sum((prices_1m - fut_price_1m) ** 2)
            res += jnp.sum((prices_3m - fut_price_3m) ** 2)
            res += 0.1 * 1e4 * jnp.sum(jnp.diff(knot_values) ** 2)
            res += 0.5 * 1e4 * (last_fixing - knot_values[0]) ** 2
            return res

        # Use jax lbfgsb to minimize with jit and autodiff
        lbfgsb = ScipyBoundedMinimize(fun=futures_objective_function, method="l-bfgs-b", jit=True)
        lower_bounds = jnp.zeros_like(initial_values)
        upper_bounds = jnp.ones_like(initial_values) * 0.1
        bounds = (lower_bounds, upper_bounds)
        res = lbfgsb.run(initial_values, bounds=bounds).params
        self.future_knot_values = res
        return self

    def plot_future_daily_forwards(self):
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        # Assuming self.future_knot_values and self.future_knot_dates are defined elsewhere
        df = 1e2 * pd.DataFrame(np.array(self.future_knot_values), index=self.future_knot_dates)

        # Create a dotted black line plot with step interpolation (left-continuous)
        plt.step(df.index, df.iloc[:, 0], where='post', linestyle=':', color='black')

        # Add black solid dots at the actual dates and values
        plt.scatter(df.index, df.iloc[:, 0], color='black', s=10, zorder=5)

        # Get current axis
        ax = plt.gca()

        # Set the y-axis limits
        ax.set_ylim(3.0, 5.25)

        # Set major ticks at multiples of 0.25
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.25))

        # Ensure the y-axis labels are shown at multiples of 0.25
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: '{:.2f}'.format(val)))

        # Format the x-axis to show date as "YY' MMM" (e.g., "24' Oct")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%y' %b"))

        # Ensure that every month has a tick/label
        ax.xaxis.set_major_locator(mdates.MonthLocator())

        # Set the x-axis labels to be vertical
        plt.xticks(rotation=90, ha='center')

        # Make the gridlines more transparent (increase transparency with alpha)
        ax.grid(which='major', axis='y', linestyle='-', linewidth='0.5', color='gray', alpha=0.3)

        # Optional: Add labels and title
        plt.xlabel('Date')
        plt.ylabel('SOFR Daily Forwards')
        plt.title('Constant Meeting-to-Meeting SOFR Daily Forwards Curve')

        # Show the plot
        plt.tight_layout()
        plt.show()

        return self

    #
    # def build_curve(self):
    #     """
    #     Build the USD SOFR curve by calibrating to market instruments.
    #     """
    #     # Initial guess for the optimization
    #     initial_rates = [0.02] * len(self.swap_knot_values)
    #     # Bounds for the rates
    #     bounds = (0.0, 0.20)
    #
    #     # Optimize to minimize the difference between market and model prices
    #     result = least_squares(self.objective_function, initial_rates, bounds=bounds)
    #
    #     # Extract the calibrated zero rates
    #     self.swap_knot_values = result.x
    #
    #     # Build discount factors and forward rates
    #     self.construct_discount_curve()
    #
    # def objective_function(self, rates):
    #     """
    #     Objective function for optimization.
    #
    #     Parameters:
    #     - rates: list of zero rates
    #
    #     Returns:
    #     - residuals: list of differences between market and model prices
    #     """
    #     residuals = []
    #
    #     idx = 0
    #     # 1M SOFR Futures
    #     for ticker, market_price in self.market_instruments['1M_Futures']:
    #         start_date, end_date = self.futures_utility.get_future_dates(ticker)
    #         model_price = self.price_sofr_future(start_date, end_date, rates[idx])
    #         residuals.append(model_price - market_price)
    #         idx += 1
    #
    #     # 3M SOFR Futures
    #     for ticker, market_price in self.market_instruments['3M_Futures']:
    #         start_date, end_date = self.futures_utility.get_future_dates(ticker)
    #         adjustment = self.convexity_adjustment(start_date)
    #         model_price = self.price_sofr_future(start_date, end_date, rates[idx], adjustment)
    #         residuals.append(model_price - market_price)
    #         idx += 1
    #
    #     # SOFR FRAs
    #     for start_date, end_date, market_rate in self.market_instruments['FRAs']:
    #         adjustment = self.convexity_adjustment(start_date)
    #         model_rate = self.forward_rate(start_date, end_date, rates[idx], adjustment)
    #         residuals.append(model_rate - market_rate)
    #         idx += 1
    #
    #     # SOFR OIS Swaps
    #     for tenor, market_rate in self.market_instruments['OIS_Swaps']:
    #         model_rate = self.price_ois_swap(tenor, rates[idx])
    #         residuals.append(model_rate - market_rate)
    #         idx += 1
    #
    #     return residuals
    #
    # def construct_discount_curve(self):
    #     """
    #     Construct discount factors and forward rates from calibrated zero rates.
    #     """
    #     dates = [self.reference_date]
    #     discounts = [1.0]
    #
    #     for idx, rate in enumerate(self.curve_nodes):
    #         # Assuming each rate corresponds to a time interval
    #         time = (idx + 1) / 12  # Assuming monthly intervals
    #         date = self.reference_date + timedelta(days=365.25 * time)
    #         df = np.exp(-rate * time)
    #         discounts.append(df)
    #         dates.append(date)
    #
    #     self.discounts = dict(zip(dates, discounts))
    #
    #     # Calculate forward rates between dates
    #     self.forward_rates = {}
    #     for i in range(len(dates) - 1):
    #         dt = (dates[i + 1] - dates[i]).days / 365.25
    #         df1 = discounts[i]
    #         df2 = discounts[i + 1]
    #         fwd_rate = (df1 / df2 - 1) / dt
    #         self.forward_rates[dates[i]] = fwd_rate
    #
    #
    # def forward_rate(self, start_date, end_date, rate, adjustment=0.0):
    #     """
    #     Calculate the forward rate between two dates.
    #
    #     Parameters:
    #     - start_date: datetime
    #     - end_date: datetime
    #     - rate: float
    #     - adjustment: float
    #
    #     Returns:
    #     - forward_rate: float
    #     """
    #     dt = (end_date - start_date).days / 365.25
    #     forward_rate = rate + adjustment
    #     return forward_rate
    #
    # def price_ois_swap(self, tenor, rate):
    #     """
    #     Price a SOFR OIS swap.
    #
    #     Parameters:
    #     - tenor: float (in years)
    #     - rate: float
    #
    #     Returns:
    #     - swap_rate: float
    #     """
    #     # Generate payment dates
    #     payment_dates = self.schedule_generator.generate_schedule(
    #         self.reference_date, tenor, frequency='Annual', calendar=self.sifma_calendar)
    #
    #     # Calculate the fixed leg
    #     fixed_leg = sum(
    #         [self.discounts[date] * rate * self.schedule_generator.day_count_fraction(date) for date in payment_dates])
    #
    #     # Calculate the floating leg (assumed to be par at initiation)
    #     floating_leg = 1 - self.discounts[payment_dates[-1]]
    #
    #     swap_rate = floating_leg / fixed_leg
    #     return swap_rate
    #
    # def get_discount_factor(self, date):
    #     """
    #     Get the discount factor for a given date.
    #
    #     Parameters:
    #     - date: datetime
    #
    #     Returns:
    #     - discount_factor: float
    #     """
    #     return self.discounts.get(date, None)
    #
    # def get_forward_rate(self, date):
    #     """
    #     Get the forward rate for a given date.
    #
    #     Parameters:
    #     - date: datetime
    #
    #     Returns:
    #     - forward_rate: float
    #     """
    #     return self.forward_rates.get(date, None)


# Example usage
if __name__ == '__main__':
    curve = USDSOFRCurve("2024-10-01")
    sofr_1m_prices = pd.Series({
        "SERV24": 95.1525,
        "SERX24": 95.315,
        "SERZ24": 95.44,
        "SERF25": 95.61,
        "SERG25": 95.81,
        "SERH25": 95.89,
        "SERJ25": 96.02,
        "SERK25": 96.13,
        "SERM25": 96.21,
        "SERN25": 96.305,
        "SERQ25": 96.39,
        "SERU25": 96.415,
    }, name="SOFR1M")
    sofr_3m_prices = pd.Series({
        "SFRU24": 95.21,
        "SFRZ24": 95.705,
        "SFRH25": 96.09,
        "SFRM25": 96.35,
        "SFRU25": 96.515,
        "SFRZ25": 96.605,
        "SFRH26": 96.655,
        "SFRM26": 96.67,
        "SFRU26": 96.67
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
    curve.load_market_data(sofr_3m_prices, sofr_1m_prices, sofr_swaps_rates)
    curve.build_future_curve()
    curve.plot_future_daily_forwards()
    exit(0)
