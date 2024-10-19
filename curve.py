import datetime as dt
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import minimize, Bounds
from scipy.interpolate import CubicSpline

from date_util import _SIFMA_, convert_date, parse_date, generate_fomc_meeting_dates, time_it
from fixing import _SOFR_
from swap import SOFRSwap
from future import IRFuture


# Pricing functions
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

def calculate_stub_fixing(ref_date: float, start_end_dates: np.ndarray, multiplicative=False) -> np.ndarray:
    """
    This function calculates the stub fixing
    :param multiplicative:
    :param ref_date:
    :param start_end_dates:
    :return:
    """
    res = np.ones((start_end_dates.shape[0]), ) if multiplicative else np.zeros((start_end_dates.shape[0]), )
    for i in range(start_end_dates.shape[0]):
        start, end = start_end_dates[i, :]
        if start >= ref_date:
            pass
        fixings = 1e-2 * _SOFR_.get_fixings_asof(parse_date(start), parse_date(ref_date-1))
        if multiplicative:
            res[i] = (1 + fixings / 360.0).prod()
        else:
            res[i] = fixings.sum()
    return res

def _price_1m_futures(future_knot_values: np.ndarray,
                      overlap_matrix: np.ndarray,
                      stub_fixings: np.ndarray,
                      n_days: np.ndarray) -> np.ndarray:
    """
    This function calculates 1m future prices. Low level numpy function.
    :param n_days:
    :param overlap_matrix:
    :param stub_fixings:
    :param future_knot_values:
    :return:
    """
    rate_sum = np.matmul(overlap_matrix, future_knot_values.reshape(-1, 1)).squeeze()
    rate_sum += stub_fixings
    return 1e2 * (1 - rate_sum / n_days)

def _price_3m_futures(future_knot_values: np.ndarray,
                      overlap_matrix: np.ndarray,
                      stub_fixings: np.ndarray,
                      n_days: np.ndarray) -> np.ndarray:
    """
    This function calculates 3m future prices. Low-level numpy function
    :param n_days:
    :param overlap_matrix:
    :param stub_fixings:
    :param future_knot_values:
    :return:
    """

    rate_prod = np.exp(np.matmul(overlap_matrix, np.log(1 + future_knot_values.reshape(-1, 1) / 360.0)).squeeze())
    rate_prod *= stub_fixings
    rate_avg = 360.0 * (rate_prod - 1) / n_days
    return 1e2 * (1 - rate_avg)

def sum_partitions(arr: np.ndarray, partitions: np.ndarray):
    """
    This function sums a lists of entries in arr according to partition given by part
    :param arr:
    :param partitions:
    :return:
    """
    indices = np.r_[0, np.cumsum(partitions)[:-1]]
    result = np.add.reduceat(arr, indices)
    return result

def _df(ref_date: float, dates: np.array, knot_dates: np.array, knot_values: np.array):
    """
    Compute df using cubic spline interpolation on zero coupon rate knots. Low level numpy function.
    :param dates:
    :param knot_dates:
    :param knot_values:
    :param ref_date:
    :return:
    """
    cs = CubicSpline(knot_dates, knot_values, extrapolate=False)
    zero_rates = cs(dates)
    zero_rates[dates < knot_dates[0]] = knot_values[0]
    zero_rates[dates > knot_dates[-1]] = knot_values[-1]
    t_vect = (dates - ref_date) / 360.0
    return np.exp(-zero_rates * t_vect)

def _price_swap_rates(swap_knot_values: np.ndarray,
                      ref_date: float,
                      swap_knot_dates: np.ndarray,
                      schedules: np.ndarray,
                      dcfs: np.ndarray,
                      partitions: np.ndarray
                      ) -> np.ndarray:
    """
    This function evaluates par rates for swaps. Low level numpy function.
    :param dcfs:
    :param ref_date:
    :param partitions:
    :param swap_knot_values:
    :param swap_knot_dates:
    :param schedules:
    :return:
    """
    dfs = _df(ref_date, schedules, swap_knot_dates, swap_knot_values)
    numerators = (dfs[:, 0] / dfs[:, 1] - 1) * dfs[:, 2]  # fwd_i * df_i
    numerators = sum_partitions(numerators, partitions)
    denominators = dcfs * dfs[:, 2]  # dcf_i * df_i
    denominators = sum_partitions(denominators, partitions)
    rates = 1e2 * numerators / denominators
    return rates

# For discrete OIS compounding df
def last_published_value(reference_dates: np.ndarray,
                         knot_dates: np.ndarray,
                         knot_values: np.ndarray) -> np.ndarray:
    """
    This function looks up reference_values for reference_dates according to knot_dates, knot_values
    :param reference_dates:
    :param knot_dates:
    :param knot_values:
    :return:
    """
    indices = np.searchsorted(knot_dates, reference_dates, side='right') - 1
    indices = np.clip(indices, 0, len(knot_values) - 1)
    return knot_values[indices]

def sofr_compound(reference_dates: np.ndarray,
                  reference_rates: np.ndarray):
    """
    This function computes the compounded SOFR rate given the fixing
    :param reference_dates:
    :param reference_rates:
    :return:
    """
    num_days = np.diff(reference_dates)
    rates = reference_rates[:-1]
    annualized_rate = np.prod(1 + rates * num_days / 360) - 1
    return 360 * annualized_rate / num_days.sum()

# Define SOFR curve class
class SOFRCurve:
    def __init__(self, reference_date):
        self.reference_date = pd.Timestamp(reference_date).to_pydatetime()
        self.market_instruments = {}
        self.future_knot_dates = None
        self.future_knot_values = None
        self.swap_knot_dates = None
        self.swap_knot_values = None
        self.convexity = None

    def plot_futures_daily_forwards(self, n_meetings=16, n_cuts=6):
        """
        Plots the future daily forward rates with additional annotations for the first n cuts.
        """
        # Assuming self.future_knot_values and self.future_knot_dates are defined elsewhere
        ind = parse_date(self.future_knot_dates[:n_meetings])
        val = 1e2 * self.future_knot_values[:n_meetings]
        fwd = pd.DataFrame(val, index=ind)

        # Create a dotted black line plot with step interpolation (left-continuous)
        plt.step(fwd.index, fwd.iloc[:, 0], where='post', linestyle=':', color='black')
        plt.scatter(fwd.index, fwd.iloc[:, 0], color='black', s=10, zorder=5)
        ax = plt.gca()
        ax.set_ylim(3.0, 5.25)
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.25))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, pos: '{:.2f}'.format(v)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%y' %b"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=90, ha='center')
        ax.grid(which='major', axis='y', linestyle='-', linewidth='0.5', color='gray', alpha=0.3)
        plt.xlabel('Date')
        plt.ylabel('SOFR Daily Forwards (%)')
        plt.title('Constant Meeting-to-Meeting SOFR Daily Forwards Curve')

        # Adding annotations for the first six step-differences
        step_diffs = 1e2 * np.diff(fwd.iloc[:n_cuts+1, 0])  # Calculate the differences for the first 4 steps
        for i in range(n_cuts):
            x_pos = fwd.index[i + 1]
            y_pos = fwd.iloc[i + 1, 0]
            plt.annotate(f"{step_diffs[i]:.1f} bps", xy=(x_pos, y_pos),
                         xytext=(x_pos, y_pos + 0.05),  # Offset annotation slightly for clarity
                         fontsize=9, color='blue')

        plt.tight_layout()
        plt.show()
        return self

    def discount_factor(self, dates: np.array) -> np.array:
        return _df(convert_date(self.reference_date), dates, self.swap_knot_dates, self.swap_knot_values)

    def future_discount_factor(self, date: dt.datetime | dt.date) -> float:
        """
        Use futures staircase to compounds forward OIS style.
        :param date:
        :return:
        """
        biz_date_range = convert_date(_SIFMA_.biz_date_range(self.reference_date, date))
        fwds = last_published_value(biz_date_range, self.future_knot_dates, self.future_knot_values)
        num_days = np.diff(biz_date_range)
        rates = fwds[:-1]
        df = 1 / np.prod(1 + rates * num_days / 360)
        return df

    def plot_swap_zero_rate(self):
        ind = parse_date(self.swap_knot_dates)
        val = 1e2 * self.swap_knot_values
        zr = pd.DataFrame(val, index=ind)
        plt.plot(zr.index, zr.values)
        ax = plt.gca()
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.25))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, pos: '{:.2f}'.format(v)))
        ax.grid(which='major', axis='y', linestyle='-', linewidth='0.5', color='gray', alpha=0.3)
        plt.xlabel('Date')
        plt.ylabel('Zero Rates (%)')
        plt.title('Continuously Compounded Zero Coupon Rate Curve')
        plt.tight_layout()
        plt.show()
        return self

    def plot_convexity(self):
        """
        This function plots future - FRA convexity
        :return:
        """
        plt.plot(self.convexity.index, 1e2 * self.convexity.values)
        ax = plt.gca()

        # Set major ticks for March, June, September, and December
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[3, 6, 9, 12]))

        # Formatting the x-axis as year and month in your preferred format
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%y' %b"))

        # Set y-axis formatting
        ax.yaxis.set_major_locator(plt.MultipleLocator(1))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, pos: '{:.2f}'.format(v)))

        # Rotate x-axis labels
        plt.xticks(rotation=90, ha='center')

        # Labels and title
        plt.xlabel('Date')
        plt.ylabel('Rate Difference (bps)')
        plt.title('Futures-FRA Convexity of 3M Forward Rates')

        plt.tight_layout()
        plt.show()

        return self

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

        self.future_knot_dates = convert_date(knot_dates)
        self.future_knot_values = 0.05 * np.ones((len(knot_dates), ))
        return self

    def initialize_swap_knots(self, swaps: list[SOFRSwap]):
        """
        This function initializes the swap curve zero rate knots given by calibrating swap instruments
        :param swaps:
        :return:
        """
        # Initialize knots for swaps
        next_biz_day = _SIFMA_.next_biz_day(self.reference_date, 0)
        knot_dates = [swap.maturity_date for swap in swaps]
        knot_dates = np.array([next_biz_day] + knot_dates)
        self.swap_knot_dates = convert_date(knot_dates)
        self.swap_knot_values = 0.05 * np.ones((len(knot_dates), ))
        return self

    @time_it
    def calibrate_swap_curve(self, spot_rates: pd.Series):
        """
        This function calibrates a swap curve's zero rate knots to prices swaps according to an input market
        :param spot_rates:
        :return:
        """
        ref_date = convert_date(self.reference_date)
        swaps = [SOFRSwap(self.reference_date, tenor=x) for x in spot_rates.index]
        mkt_rates = spot_rates.values.squeeze()

        self.initialize_swap_knots(swaps)

        # Now we generate and merge the schedule array into a huge one with partition recorded.
        schedules = [swap.get_float_leg_schedule(True).values for swap in swaps]
        partition = np.array([len(x) for x in schedules])
        schedule_block  = np.concatenate(schedules, axis=0)
        schedules = schedule_block[:, :-1]
        dcfs = schedule_block[:, -1].squeeze()

        knot_dates = self.swap_knot_dates
        initial_values = 0.05 * np.ones_like(self.swap_knot_values)

        def loss_function(knot_values: np.array) -> float:
            rates = _price_swap_rates(knot_values, ref_date, knot_dates, schedules, dcfs, partition)
            loss = np.sum((rates - mkt_rates) ** 2)
            loss += 1e2 * np.sum(np.diff(knot_values) ** 2)
            return loss

        bounds = Bounds(0.0, 0.08)
        res = minimize(loss_function,
                       initial_values,
                       method="L-BFGS-B",
                       bounds=bounds)

        # Set curve status
        self.swap_knot_values = res.x
        self.market_instruments["SOFRSwaps"] = spot_rates
        return self

    @time_it
    def calibrate_futures_curve(self, futures_1m_prices: pd.Series, futures_3m_prices: pd.Series):
        """
        This function calibrates the futures curve to the 1m and 3m futures prices
        :param futures_1m_prices:
        :param futures_3m_prices:
        :return:
        """
        # Create the futures
        ref_date = convert_date(self.reference_date)
        futures_1m = [IRFuture(x) for x in futures_1m_prices.index]
        futures_3m = [IRFuture(x) for x in futures_3m_prices.index]
        px_1m = futures_1m_prices.values.squeeze()
        px_3m = futures_3m_prices.values.squeeze()

        # Initialize the FOMC meeting effective dates up till expiry of the last 3m future
        self.initialize_future_knots(futures_3m[-1].reference_end_date)

        # Obtain the start and end dates of the 1m and 3m futures
        fut_start_end_1m = np.array([convert_date(fut.get_reference_start_end_dates()) for fut in futures_1m])
        days_1m = (np.diff(fut_start_end_1m, axis=1) + 1).squeeze()
        fut_start_end_3m = np.array([convert_date(fut.get_reference_start_end_dates()) for fut in futures_3m])
        days_3m = (np.diff(fut_start_end_3m, axis=1) + 1).squeeze()

        # stubs
        stubs_1m = calculate_stub_fixing(ref_date, fut_start_end_1m, False)
        stubs_3m = calculate_stub_fixing(ref_date, fut_start_end_3m, True)

        # Create the overlap matrices for 1m and 3m futures for knot periods
        o_matrix_1m = create_overlap_matrix(fut_start_end_1m, self.future_knot_dates).astype(float)
        o_matrix_3m = create_overlap_matrix(fut_start_end_3m, self.future_knot_dates).astype(float)

        def objective_function(knot_values: np.ndarray,
                               o_mat_1m: np.ndarray, fixing_1m: np.ndarray, n_days_1m: np.ndarray, prices_1m: np.ndarray,
                               o_mat_3m: np.ndarray, fixing_3m: np.ndarray, n_days_3m: np.ndarray, prices_3m: np.ndarray):
            res_1m = _price_1m_futures(knot_values, o_mat_1m, fixing_1m, n_days_1m)
            loss = np.sum((res_1m - prices_1m) ** 2)
            res_3m = _price_3m_futures(knot_values, o_mat_3m, fixing_3m, n_days_3m)
            loss += np.sum((res_3m - prices_3m) ** 2)
            # Add a little curve constraint loss
            loss += 1e2 * np.sum(np.diff(knot_values) ** 2)
            return loss

        # Initial values
        initial_knot_values = self.future_knot_values
        bounds = Bounds(0.0, 0.08)
        res = minimize(objective_function,
                       initial_knot_values,
                       args=(o_matrix_1m, stubs_1m, days_1m, px_1m,
                             o_matrix_3m, stubs_3m, days_3m, px_3m),
                       method="L-BFGS-B",
                       bounds=bounds)

        # Set curve status
        self.future_knot_values = res.x
        self.market_instruments["SOFR1M"] = futures_1m_prices
        self.market_instruments["SOFR3M"] = futures_3m_prices
        return self

    @time_it
    def calibrate_futures_curve_3m(self, futures_3m_prices: pd.Series):
        """
        This function calibrates the futures curve to the 1m and 3m futures prices
        :param futures_3m_prices:
        :return:
        """
        # Create the futures
        ref_date = convert_date(self.reference_date)
        futures_3m = [IRFuture(x) for x in futures_3m_prices.index]
        px_3m = futures_3m_prices.values.squeeze()

        # Initialize the FOMC meeting effective dates up till expiry of the last 3m future
        self.initialize_future_knots(futures_3m[-1].reference_end_date)


        fut_start_end_3m = np.array([convert_date(fut.get_reference_start_end_dates()) for fut in futures_3m])
        days_3m = (np.diff(fut_start_end_3m, axis=1) + 1).squeeze()
        stubs_3m = calculate_stub_fixing(ref_date, fut_start_end_3m, True)
        o_matrix_3m = create_overlap_matrix(fut_start_end_3m, self.future_knot_dates).astype(float)

        def objective_function(knot_values: np.ndarray,
                               o_mat_3m: np.ndarray, fixing_3m: np.ndarray, n_days_3m: np.ndarray,
                               prices_3m: np.ndarray):
            res_3m = _price_3m_futures(knot_values, o_mat_3m, fixing_3m, n_days_3m)
            loss = np.sum((res_3m - prices_3m) ** 2)
            # Add a little curve constraint loss
            loss += 1e2 * np.sum(np.diff(knot_values) ** 2)
            return loss

        # Initial values
        initial_knot_values = self.future_knot_values
        bounds = Bounds(0.0, 0.08)
        res = minimize(objective_function,
                       initial_knot_values,
                       args=(o_matrix_3m, stubs_3m, days_3m, px_3m),
                       method="L-BFGS-B",
                       bounds=bounds)

        # Set curve status
        self.future_knot_values = res.x
        self.market_instruments["SOFR3M"] = futures_3m_prices
        return self

    @time_it
    def calculate_convexity(self):
        """
        This function uses curve to evaluate future prices as well as equivalent swap rates.
        :return:
        """
        fut_3m = self.market_instruments["SOFR3M"]
        fut_rates = 1e2 - fut_3m.values.squeeze()[1:]
        fut_st_et = [IRFuture(x).get_reference_start_end_dates() for x in fut_3m.index[1:]]
        fra = [SOFRSwap(self.reference_date, x, y + dt.timedelta(days=1)) for x, y in fut_st_et]
        fra_rates = price_swap_rates(self, fra)
        df = pd.DataFrame(fut_rates - fra_rates, index=pd.DatetimeIndex([x[0] for x in fut_st_et]))
        self.convexity = df
        return self


# External pricing functionalities
@time_it
def price_1m_futures(curve: SOFRCurve, futures_1m: list[str] | np.ndarray[str]) -> np.ndarray:
    """
    This function prices a list of SOFR 1m futures on a curve
    :param curve:
    :param futures_1m:
    :return:
    """
    ref_date = convert_date(curve.reference_date)
    st_et = np.array([convert_date(IRFuture(x).get_reference_start_end_dates()) for x in futures_1m])
    o_matrix = create_overlap_matrix(st_et, curve.future_knot_dates)
    stubs = calculate_stub_fixing(ref_date, st_et, False)
    n_days = (np.diff(st_et, axis=1) + 1).squeeze()
    return _price_1m_futures(curve.future_knot_values, o_matrix, stubs, n_days)

@time_it
def price_3m_futures(curve: SOFRCurve, futures_3m: list[str] | np.ndarray[str]) -> np.ndarray:
    """
    This function prices a list of SOFR 3m futures on a curve
    :param curve:
    :param futures_3m:
    :return:
    """
    ref_date = convert_date(curve.reference_date)
    st_et = np.array([convert_date(IRFuture(x).get_reference_start_end_dates()) for x in futures_3m])
    o_matrix = create_overlap_matrix(st_et, curve.future_knot_dates)
    stubs = calculate_stub_fixing(ref_date, st_et, True)
    n_days = (np.diff(st_et, axis=1) + 1).squeeze()
    return _price_3m_futures(curve.future_knot_values, o_matrix, stubs, n_days)

@time_it
def price_swap_rates(curve: SOFRCurve, swaps: list[SOFRSwap]) -> np.ndarray:
    """
    This function prices a list of SOFR swaps
    :param curve:
    :param swaps:
    :return:
    """
    ref_date = convert_date(curve.reference_date)
    schedules = [swap.get_float_leg_schedule(True).values for swap in swaps]
    partition = np.array([len(x) for x in schedules])
    schedule_block = np.concatenate(schedules, axis=0)
    schedules = schedule_block[:, :-1]
    dcfs = schedule_block[:, -1].squeeze()
    return _price_swap_rates(curve.swap_knot_values,
                      ref_date,
                      curve.swap_knot_dates,
                      schedules,
                      dcfs,
                      partition
                      )

def price_spot_rates(curve: SOFRCurve, tenors: list[str] | np.ndarray[str]) -> np.ndarray:
    """
    This function prices a list of spot starting par rates given tenor
    :param curve:
    :param tenors:
    :return:
    """
    swaps = [SOFRSwap(curve.reference_date, tenor=x) for x in tenors]
    return price_swap_rates(curve, swaps)

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

    sofr = SOFRCurve("2024-10-09")
    sofr.calibrate_futures_curve(sofr_1m_prices, sofr_3m_prices)
    sofr.calibrate_swap_curve(sofr_swaps_rates)
    sofr.calculate_convexity()

    print("Pricing errors in bps for SOFR1M futures:")
    print(1e2 * (price_1m_futures(sofr, sofr_1m_prices.index) - sofr_1m_prices.values))

    print("Pricing errors in bps for SOFR3M futures:")
    print(1e2 * (price_3m_futures(sofr, sofr_3m_prices.index) - sofr_3m_prices.values))

    print("Pricing errors in bps for swaps")
    print(1e2 * (price_spot_rates(sofr, sofr_swaps_rates.index) - sofr_swaps_rates.values))

    sofr.plot_futures_daily_forwards(16, 6)
    sofr.plot_swap_zero_rate()
    sofr.plot_convexity()
    exit(0)
