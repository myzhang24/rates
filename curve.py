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
def _create_overlap_matrix(start_end_dates: np.ndarray,
                           knot_dates: np.ndarray) -> np.ndarray:
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

def _calculate_stub_fixing(ref_date: float,
                           start_end_dates: np.ndarray,
                           multiplicative=False) -> (float, np.ndarray):
    """
    This function calculates the stub fixing. Returns overnight rate and the stub fixing sum or accrual
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
    on = 1e-2 * _SOFR_.get_fixings_asof(parse_date(ref_date-4), parse_date(ref_date-1)).iloc[-1]
    return on, res

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

def _sum_partitions(arr: np.ndarray, partitions: np.ndarray):
    """
    This function sums a lists of entries in arr according to partition given by part
    :param arr:
    :param partitions:
    :return:
    """
    indices = np.r_[0, np.cumsum(partitions)[:-1]]
    result = np.add.reduceat(arr, indices)
    return result

def _df(ref_date: float, dates: np.ndarray, knot_dates: np.ndarray, knot_values: np.ndarray):
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
    numerators = _sum_partitions(numerators, partitions)
    denominators = dcfs * dfs[:, 2]  # dcf_i * df_i
    denominators = _sum_partitions(denominators, partitions)
    rates = 1e2 * numerators / denominators
    return rates

# For discrete OIS compounding df
def _last_published_value(reference_dates: np.ndarray,
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

def _ois_compound(reference_dates: np.ndarray,
                  reference_rates: np.ndarray):
    """
    This function computes the compounded OIS rate given the fixing
    :param reference_dates:
    :param reference_rates:
    :return:
    """
    num_days = np.diff(reference_dates)
    rates = reference_rates[:-1]
    annualized_rate = np.prod(1 + rates * num_days / 360) - 1
    return 360 * annualized_rate / num_days.sum()

########################################################################################################################
# Define USD curve class
########################################################################################################################
class USDCurve:
    def __init__(self, reference_date):
        self.reference_date = pd.Timestamp(reference_date).to_pydatetime()
        self.is_fomc = False    # Whether today is an FOMC effective date (one biz day after second meeting date)
        self.market_data = {}
        self.fomc_effective_dates = None
        self.effective_rates_sofr = None
        self.effective_rates_ff = None
        self.swap_knot_dates = None
        self.swap_knot_values = None
        self.sofr_ff_spread = None
        self.sofr_future_swap_spread = None

    def plot_effective_rates(self, rate="sofr", n_meetings=16, n_cuts=6):
        """
        Plots the future implied daily forwards for various FOMC meeting effective dates.
        :param rate:
        :param n_meetings:
        :param n_cuts:
        :return:
        """
        # Assuming self.future_knot_values and self.future_knot_dates are defined elsewhere
        ind = parse_date(self.fomc_effective_dates[:n_meetings])
        val = self.effective_rates_sofr[:n_meetings] if rate.lower() == "sofr" else self.effective_rates_ff[:n_meetings]
        val *= 1e2
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
        plt.ylabel(f'{rate.upper()} Daily Forwards (%)')
        plt.title(f'Constant Meeting-to-Meeting {rate.upper()} Daily Forwards Curve')

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

    def swap_discount_factor(self, dates: np.ndarray) -> np.ndarray:
        """
        This function returns the discount factor according to the sofr swap curve
        :param dates:
        :return:
        """
        return _df(convert_date(self.reference_date), dates, self.swap_knot_dates, self.swap_knot_values)

    def forward_rates(self, st: np.ndarray, et: np.ndarray) -> np.ndarray:
        """
        Returns the annualized forward rates from st to et (not accruing over et to et+1)
        :param st:
        :param et:
        :return:
        """
        return 360 * (self.swap_discount_factor(st) / self.swap_discount_factor(et) - 1) / (et - st)

    def future_discount_factor(self, rate: str, date: dt.datetime | dt.date) -> float:
        """
        Use futures staircase to compounds forward OIS style.
        :param rate:
        :param date:
        :return:
        """
        biz_date_range = convert_date(_SIFMA_.biz_date_range(self.reference_date, date))
        effective_rates = self.effective_rates_sofr if rate.lower() == "sofr" else self.effective_rates_ff
        fwds = _last_published_value(biz_date_range, self.fomc_effective_dates, effective_rates)
        num_days = np.diff(biz_date_range)
        rates = fwds[:-1]
        df = 1 / np.prod(1 + rates * num_days / 360)
        return df

    def plot_swap_zero_rates(self):
        """
        This function plots the swap zero rates curve
        :return:
        """
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
            self.is_fomc = True

        self.fomc_effective_dates = convert_date(knot_dates)
        self.effective_rates_sofr = 0.05 * np.ones((len(knot_dates),))
        self.effective_rates_ff = 0.05 * np.ones((len(knot_dates),))
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
        self.market_data["SOFRSwaps"] = spot_rates
        return self

    @time_it
    def calibrate_future_curve_sofr(self, futures_1m_prices: pd.Series, futures_3m_prices: pd.Series):
        """
        This function calibrates the futures curve to the 1m and 3m futures prices
        :param futures_1m_prices:
        :param futures_3m_prices:
        :return:
        """
        # Create the futures
        ref_date = convert_date(self.reference_date)
        fomc = 1e-2 if self.is_fomc else 1.0
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
        on, stubs_1m = _calculate_stub_fixing(ref_date, fut_start_end_1m, False)
        _, stubs_3m = _calculate_stub_fixing(ref_date, fut_start_end_3m, True)

        # Create the overlap matrices for 1m and 3m futures for knot periods
        o_matrix_1m = _create_overlap_matrix(fut_start_end_1m, self.fomc_effective_dates).astype(float)
        o_matrix_3m = _create_overlap_matrix(fut_start_end_3m, self.fomc_effective_dates).astype(float)

        def objective_function(knot_values: np.ndarray,
                               o_mat_1m: np.ndarray, fixing_1m: np.ndarray, n_days_1m: np.ndarray, prices_1m: np.ndarray,
                               o_mat_3m: np.ndarray, fixing_3m: np.ndarray, n_days_3m: np.ndarray, prices_3m: np.ndarray):
            res_1m = _price_1m_futures(knot_values, o_mat_1m, fixing_1m, n_days_1m)
            loss = np.sum((res_1m - prices_1m) ** 2)
            res_3m = _price_3m_futures(knot_values, o_mat_3m, fixing_3m, n_days_3m)
            loss += np.sum((res_3m - prices_3m) ** 2)
            # Add a little curve constraint loss
            loss += 1e2 * np.sum(np.diff(knot_values) ** 2)
            loss += fomc * 1e4 * (on - knot_values[0]) ** 2
            return loss

        # Initial values
        initial_knot_values = self.effective_rates_sofr
        bounds = Bounds(0.0, 0.08)
        res = minimize(objective_function,
                       initial_knot_values,
                       args=(o_matrix_1m, stubs_1m, days_1m, px_1m,
                             o_matrix_3m, stubs_3m, days_3m, px_3m),
                       method="L-BFGS-B",
                       bounds=bounds)

        # Set curve status
        self.effective_rates_sofr = res.x
        self.market_data["SOFR1M"] = futures_1m_prices
        self.market_data["SOFR3M"] = futures_3m_prices
        return self

    @time_it
    def calibrate_future_curve_sofr3m(self, futures_3m_prices: pd.Series):
        """
        This function calibrates the futures curve to the 3m futures prices
        :param futures_3m_prices:
        :return:
        """
        # Create the futures
        ref_date = convert_date(self.reference_date)
        fomc = 1e-2 if self.is_fomc else 1.0
        futures_3m = [IRFuture(x) for x in futures_3m_prices.index]
        px_3m = futures_3m_prices.values.squeeze()

        # Initialize the FOMC meeting effective dates up till expiry of the last 3m future
        self.initialize_future_knots(futures_3m[-1].reference_end_date)


        fut_start_end_3m = np.array([convert_date(fut.get_reference_start_end_dates()) for fut in futures_3m])
        days_3m = (np.diff(fut_start_end_3m, axis=1) + 1).squeeze()
        on, stubs_3m = _calculate_stub_fixing(ref_date, fut_start_end_3m, True)
        o_matrix_3m = _create_overlap_matrix(fut_start_end_3m, self.fomc_effective_dates).astype(float)

        def objective_function(knot_values: np.ndarray,
                               o_mat_3m: np.ndarray, fixing_3m: np.ndarray, n_days_3m: np.ndarray,
                               prices_3m: np.ndarray):
            res_3m = _price_3m_futures(knot_values, o_mat_3m, fixing_3m, n_days_3m)
            loss = np.sum((res_3m - prices_3m) ** 2)
            # Add a little curve constraint loss
            loss += 1e2 * np.sum(np.diff(knot_values) ** 2)
            loss += fomc * 1e4 * (on - knot_values[0]) ** 2
            return loss

        # Initial values
        initial_knot_values = self.effective_rates_sofr
        bounds = Bounds(0.0, 0.08)
        res = minimize(objective_function,
                       initial_knot_values,
                       args=(o_matrix_3m, stubs_3m, days_3m, px_3m),
                       method="L-BFGS-B",
                       bounds=bounds)

        # Set curve status
        self.effective_rates_sofr = res.x
        self.market_data["SOFR3M"] = futures_3m_prices
        return self

    @time_it
    def calibrate_future_curve_1m(self, rate: str, futures_1m_prices: pd.Series):
        """
        This function calibrates the futures curve to the 1m futures prices
        :param futures_1m_prices:
        :return:
        """
        # Create the futures
        ref_date = convert_date(self.reference_date)
        fomc = 1e-2 if self.is_fomc else 1.0
        futures_1m = [IRFuture(x) for x in futures_1m_prices.index]
        px_1m = futures_1m_prices.values.squeeze()

        # Initialize the FOMC meeting effective dates up till expiry of the last 3m future
        self.initialize_future_knots(futures_1m[-1].reference_end_date)

        # Obtain the start and end dates of the 1m and 3m futures
        fut_start_end_1m = np.array([convert_date(fut.get_reference_start_end_dates()) for fut in futures_1m])
        days_1m = (np.diff(fut_start_end_1m, axis=1) + 1).squeeze()

        # stubs
        on, stubs_1m = _calculate_stub_fixing(ref_date, fut_start_end_1m, False)
        o_matrix_1m = _create_overlap_matrix(fut_start_end_1m, self.fomc_effective_dates).astype(float)

        def objective_function(knot_values: np.ndarray,
                               o_mat_1m: np.ndarray, fixing_1m: np.ndarray, n_days_1m: np.ndarray,
                               prices_1m: np.ndarray
        ):
            res_1m = _price_1m_futures(knot_values, o_mat_1m, fixing_1m, n_days_1m)
            loss = np.sum((res_1m - prices_1m) ** 2)
            # Add a little curve constraint loss
            loss += 1e2 * np.sum(np.diff(knot_values) ** 2)
            loss += fomc * 1e4 * (on - knot_values[0]) ** 2
            return loss

        # Initial values
        initial_knot_values = self.effective_rates_sofr
        bounds = Bounds(0.0, 0.08)
        res = minimize(objective_function,
                       initial_knot_values,
                       args=(o_matrix_1m, stubs_1m, days_1m, px_1m),
                       method="L-BFGS-B",
                       bounds=bounds)

        # Set curve status
        if rate.lower() == "sofr":
            self.effective_rates_sofr = res.x
            self.market_data["SOFR1M"] = futures_1m_prices
        if rate.lower() == "ff":
            self.effective_rates_ff = res.x
            self.market_data["FF"] = futures_1m_prices
        return self

    @time_it
    def calculate_sofr_future_swap_convexity(self):
        """
        This function uses curve to evaluate future prices as well as equivalent swap rates.
        :return:
        """
        fut_3m = self.market_data["SOFR3M"]
        fut_rates = 1e2 - fut_3m.values.squeeze()[1:]
        fut_st_et = np.array([convert_date(IRFuture(x).get_reference_start_end_dates()) for x in fut_3m.index[1:]])
        fut_st_et[:, 1] += 1    # This is because the end day is one day before IMM, which needs to be accrued.
        forward_rates = 1e2 * self.forward_rates(fut_st_et[:, 0], fut_st_et[:, 1])
        df = pd.Series(fut_rates - forward_rates, index=pd.DatetimeIndex(parse_date(fut_st_et[:, 0])))
        self.sofr_future_swap_spread = df
        return self

    @time_it
    def calculate_sofr_ff_spread(self):
        """
        This function uses curve to evaluate future prices as well as equivalent swap rates.
        :return:
        """
        sofr = self.market_data["SOFR1M"]
        ff = self.market_data["FF"]
        fut_st_et = np.array([convert_date(IRFuture(x).get_reference_start_end_dates()) for x in sofr.index])
        df = pd.Series(ff.values.squeeze() - sofr.values.squeeze(), index=pd.DatetimeIndex(parse_date(fut_st_et[:, 0])))
        self.sofr_ff_spread = df
        return self


# External pricing functionalities
@time_it
def price_1m_futures(curve: USDCurve, futures_1m: list[str] | np.ndarray[str]) -> np.ndarray:
    """
    This function prices a list of SOFR 1m futures on a curve
    :param curve:
    :param futures_1m:
    :return:
    """
    ref_date = convert_date(curve.reference_date)
    st_et = np.array([convert_date(IRFuture(x).get_reference_start_end_dates()) for x in futures_1m])
    o_matrix = _create_overlap_matrix(st_et, curve.fomc_effective_dates)
    _, stubs = _calculate_stub_fixing(ref_date, st_et, False)
    n_days = (np.diff(st_et, axis=1) + 1).squeeze()
    return _price_1m_futures(curve.effective_rates_sofr, o_matrix, stubs, n_days)

@time_it
def price_3m_futures(curve: USDCurve, futures_3m: list[str] | np.ndarray[str]) -> np.ndarray:
    """
    This function prices a list of SOFR 3m futures on a curve
    :param curve:
    :param futures_3m:
    :return:
    """
    ref_date = convert_date(curve.reference_date)
    st_et = np.array([convert_date(IRFuture(x).get_reference_start_end_dates()) for x in futures_3m])
    o_matrix = _create_overlap_matrix(st_et, curve.fomc_effective_dates)
    _, stubs = _calculate_stub_fixing(ref_date, st_et, True)
    n_days = (np.diff(st_et, axis=1) + 1).squeeze()
    return _price_3m_futures(curve.effective_rates_sofr, o_matrix, stubs, n_days)

@time_it
def price_swap_rates(curve: USDCurve, swaps: list[SOFRSwap]) -> np.ndarray:
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

def price_spot_rates(curve: USDCurve, tenors: list[str] | np.ndarray[str]) -> np.ndarray:
    """
    This function prices a list of spot starting par rates given tenor
    :param curve:
    :param tenors:
    :return:
    """
    swaps = [SOFRSwap(curve.reference_date, tenor=x) for x in tenors]
    return price_swap_rates(curve, swaps)


def debug_joint_calibration():
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

    sofr = USDCurve("2024-10-09")
    sofr.calibrate_future_curve_sofr(sofr_1m_prices, sofr_3m_prices)
    sofr.calibrate_swap_curve(sofr_swaps_rates)
    sofr.calculate_sofr_future_swap_convexity()
    print("Pricing errors in bps for SOFR1M futures:")
    print(1e2 * (price_1m_futures(sofr, sofr_1m_prices.index) - sofr_1m_prices.values))

    print("Pricing errors in bps for SOFR3M futures:")
    print(1e2 * (price_3m_futures(sofr, sofr_3m_prices.index) - sofr_3m_prices.values))

    print("Pricing errors in bps for swaps")
    print(1e2 * (price_spot_rates(sofr, sofr_swaps_rates.index) - sofr_swaps_rates.values))

    sofr.plot_effective_rates(16, 6)
    sofr.plot_swap_zero_rates()
    sofr.plot_convexity()

def debug_1m_calibration():
    ff = pd.Series({
        "FFV4": 95.17,
        "FFX4": 95.345,
        "FFZ4": 95.475,
        "FFF5": 95.605,
        "FFG5": 95.77,
        "FFH5": 95.86,
        "FFJ5": 96.00,
        "FFK5": 96.135,
        "FFM5": 96.24,
        "FFN5": 96.345,
        "FFQ5": 96.435,
        "FFU5": 96.475,
        "FFV5": 96.53,
    })
    ref_date = dt.datetime(2024, 10, 18)
    ff_curve = USDCurve(ref_date).calibrate_future_curve_1m(ff)

    ff_curve.plot_effective_rates(16, 6)
    exit(0)

# Example usage
if __name__ == '__main__':
    # debug_1m_calibration()

    debug_joint_calibration()
    exit(0)

