import datetime as dt
from copy import deepcopy
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import minimize, Bounds

from fixing import _SOFR_, _FF_
from date_util import _SIFMA_, convert_date, parse_date, generate_fomc_meeting_dates, time_it
from math_util import _df, _last_published_value, _price_swap_rates, _calculate_stub_fixing, _create_overlap_matrix, _price_1m_futures, _price_3m_futures
from swap import SOFRSwap
from future import IRFuture, live_futures


########################################################################################################################
# Define USD curve class
########################################################################################################################
class USDCurve:
    def __init__(self, rate_name: str, reference_date):
        self.rate_name = rate_name.upper()
        self.reference_date = pd.Timestamp(reference_date).to_pydatetime()
        self.is_fomc = False    # Whether today is an FOMC effective date (one biz day after second meeting date)
        self.market_data = {}

        self._fomc_effective_dates = None
        self._effective_rates = None
        self._swap_knot_dates = None
        self._swap_knot_values = None
        self._future_swap_spread = None
        self.convexity_model = None

    def reprice_futures(self):
        """
        This function reprices futures prices according to model and replace data in market_data
        :return:
        """
        if self.rate_name == "FF":
            ff_tickers = live_futures(self.reference_date, "ff")
            ff_px = np.round(self.price_1m_futures(ff_tickers), 4)
            self.market_data["FF"] = pd.Series(ff_px, index=ff_tickers)
            return self
        if self.rate_name == "SOFR":
            # SOFR3M
            sofr3m_tickers = live_futures(self.reference_date, "sofr3m")
            sofr3m_px = np.round(self.price_3m_futures(sofr3m_tickers), 4)
            self.market_data["SOFR3M"] = pd.Series(sofr3m_px, index=sofr3m_tickers)
            # SOFR1M
            sofr1m_tickers = live_futures(self.reference_date, "sofr1m")
            sofr1m_px = np.round(self.price_1m_futures(sofr1m_tickers), 4)
            self.market_data["SOFR1M"] = pd.Series(sofr1m_px, index=sofr1m_tickers)
        return self

    def reprice_swaps(self):
        """
        This function reprices swap instruments according to zero curve and replace data
        :return:
        """
        swap_tenors = self.market_data.get("SOFRSwaps", pd.Series(index=["1M", "3M", "6M", "9M", "12M",
                                                                         "18M", "2Y", "3Y", "4Y", "5Y",
                                                                         "7Y", "10Y", "12Y", "15Y", "20Y",
                                                                         "25Y", "30Y"])).index
        swap_rates = np.round(self.price_spot_rates(swap_tenors), 4)
        self.market_data["SOFRSwaps"] = pd.Series(swap_rates, index=swap_tenors)
        return self

    def shock_effective_rate(self, amount_bps, reprice=True):
        self._effective_rates += amount_bps * 1e-4
        if reprice:
            self.reprice_futures()
        return self

    def shock_zero_rate(self, amount_bps, reprice=True):
        self._swap_knot_values += amount_bps * 1e-4
        if reprice:
            self.reprice_swaps()
        return self

    def set_effective_rate(self, effective_rate: np.ndarray, reprice=True):
        assert self._effective_rates.shape == effective_rate.shape
        self._effective_rates = effective_rate
        if reprice:
            self.reprice_futures()
        return self

    def set_zero_rate(self, zero_rate: np.ndarray, reprice=True):
        assert self._swap_knot_values.shape == zero_rate.shape
        self._swap_knot_values = zero_rate
        if reprice:
            self.reprice_swaps()
        return self

    def plot_effective_rates(self, n_meetings=16, n_cuts=6):
        """
        Plots the future implied daily forwards for various FOMC meeting effective dates.
        :param n_meetings:
        :param n_cuts:
        :return:
        """
        # Assuming self.future_knot_values and self.future_knot_dates are defined elsewhere
        ind = parse_date(self._fomc_effective_dates[:n_meetings])
        val = self._effective_rates[:n_meetings]
        fwd = 1e2 * pd.DataFrame(val, index=ind)

        # Create a dotted black line plot with step interpolation (left-continuous)
        plt.figure()
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
        plt.ylabel(f'{self.rate_name.upper()} Daily Forwards (%)')
        plt.title(f'Constant Meeting-to-Meeting {self.rate_name.upper()} Daily Forwards Curve')

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
        return _df(convert_date(self.reference_date), dates, self._swap_knot_dates, self._swap_knot_values)

    def forward_rates(self, st: np.ndarray, et: np.ndarray) -> np.ndarray:
        """
        Returns the annualized forward rates from st to et (not accruing over et to et+1)
        :param st:
        :param et:
        :return:
        """
        return 360 * (self.swap_discount_factor(st) / self.swap_discount_factor(et) - 1) / (et - st)

    def future_discount_factor(self, date: dt.datetime | dt.date) -> float:
        """
        Use futures staircase to compounds forward OIS style.
        :param date:
        :return:
        """
        biz_date_range = convert_date(_SIFMA_.biz_date_range(self.reference_date, date))
        fwds = _last_published_value(biz_date_range, self._fomc_effective_dates, self._effective_rates)
        num_days = np.diff(biz_date_range)
        rates = fwds[:-1]
        df = 1 / np.prod(1 + rates * num_days / 360)
        return df

    def plot_swap_zero_rates(self):
        """
        This function plots the swap zero rates curve
        :return:
        """
        ind = parse_date(self._swap_knot_dates)
        val = 1e2 * self._swap_knot_values
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

    def plot_sofr_future_swap_spread(self):
        """
        This function plots future - FRA convexity
        :return:
        """
        plt.plot(self._future_swap_spread.index, 1e2 * self._future_swap_spread.values)
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

        self._fomc_effective_dates = convert_date(knot_dates)
        self._effective_rates = 0.05 * np.ones((len(knot_dates),))
        return self

    def initialize_swap_knots(self, swaps: list[SOFRSwap]):
        """
        This function initializes the swap curve zero rate knots given by calibrating swap instruments
        :param swaps:
        :return:
        """
        # Initialize knots for swaps
        next_biz_day = _SIFMA_.next_biz_day(self.reference_date, 0)
        knot_dates = sorted([swap.maturity_date for swap in swaps])
        knot_dates = np.array([next_biz_day] + knot_dates)
        self._swap_knot_dates = convert_date(knot_dates)
        self._swap_knot_values = 0.05 * np.ones((len(knot_dates),))
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

        knot_dates = self._swap_knot_dates
        initial_values = 0.05 * np.ones_like(self._swap_knot_values)

        def loss_function(knot_values: np.array) -> float:
            rates = _price_swap_rates(knot_values, ref_date, knot_dates, schedules, dcfs, partition)
            loss = np.sum((rates - mkt_rates) ** 2)
            loss += 1e2 * np.sum(np.diff(knot_values) ** 2)
            return loss

        bounds = Bounds(0.0, 0.08)
        res = minimize(loss_function,
                       initial_values,
                       method="L-BFGS-B",
                       tol=1e-9,
                       bounds=bounds)

        # Set curve status
        self._swap_knot_values = res.x
        self.market_data["SOFRSwaps"] = spot_rates
        return self

    @time_it
    def calibrate_swap_curve_with_convexity(self, sofr3m_futures: pd.Series, spot_rates: pd.Series):
        """
        This function calibrates a swap curve's zero rate knots to prices swaps according to an input market
        Simultaneously, we assume a linear convexity adjustment (applies to Ho-Lee model, or general Gaussian models)
        and want to match the 3m futures as well.
        :param sofr3m_futures:
        :param spot_rates:
        :return:
        """
        ref_date = convert_date(self.reference_date)
        spot_swaps = [SOFRSwap(self.reference_date, tenor=x) for x in spot_rates.index]
        fut_st_et = [IRFuture(fut).get_reference_start_end_dates() for fut in sofr3m_futures.index[1:]]
        fut_swaps = [SOFRSwap(self.reference_date, x, y + dt.timedelta(days=1)) for x, y in fut_st_et]
        swaps = fut_swaps + spot_swaps
        mkt_rates = spot_rates.values.squeeze()

        self.initialize_swap_knots(spot_swaps)

        # Now we generate and merge the schedule array into a huge one with partition recorded.
        schedules = [swap.get_float_leg_schedule(True).values for swap in swaps]
        partition = np.array([len(x) for x in schedules])
        schedule_block  = np.concatenate(schedules, axis=0)
        schedules = schedule_block[:, :-1]
        dcfs = schedule_block[:, -1].squeeze()

        knot_dates = self._swap_knot_dates
        initial_values = 0.05 * np.ones(self._swap_knot_values.shape[0] + 1) # First two are
        initial_values[0] = 0.02

        fwd_ness = 1/360 * np.array([convert_date(IRFuture(fut).reference_start_date) - ref_date for fut in sofr3m_futures.index[1:]])
        fut_rates = 1e2 - sofr3m_futures.values.squeeze()[1:]
        n_fut = len(fwd_ness)

        def loss_function(knot_values: np.array) -> float:
            rates = _price_swap_rates(knot_values[1:], ref_date, knot_dates, schedules, dcfs, partition)
            fra_rates = rates[:n_fut]
            convexity = fwd_ness * knot_values[0]
            loss = 0.5 * np.sum(((fra_rates + convexity) - fut_rates) ** 2)
            swap_rates = rates[n_fut:]
            loss += np.sum((swap_rates - mkt_rates) ** 2)
            loss += 1e2 * np.sum(np.diff(knot_values[1:]) ** 2)
            return loss

        bounds = Bounds(0.0, 0.08)
        res = minimize(loss_function,
                       initial_values,
                       method="SLSQP",
                       bounds=bounds)

        # Set curve status
        self.convexity_model = res.x[0]
        self._swap_knot_values = res.x[1:]
        self.market_data["SOFRSwaps"] = spot_rates
        self.market_data["SOFR3M"] = sofr3m_futures
        return self

    @time_it
    def calibrate_future_curve(self, futures_1m_prices: pd.Series=None, futures_3m_prices: pd.Series=None):
        """
        This function calibrates the sofr futures curve to the 1m and 3m futures prices
        :param futures_1m_prices:
        :param futures_3m_prices:
        :return:
        """
        # Create the futures
        ref_date = convert_date(self.reference_date)
        fomc = 1e-2 if self.is_fomc else 1.0
        end_date = self.reference_date
        if futures_1m_prices is not None:
            end_date = max(end_date, IRFuture(futures_1m_prices.index[-1]).reference_end_date)
        if futures_3m_prices is not None:
            end_date = max(end_date, IRFuture(futures_3m_prices.index[-1]).reference_end_date)
        assert end_date > self.reference_date
        self.initialize_future_knots(end_date)

        # Null initiation
        def objective_function_1m(knot_values: np.ndarray,
                                  o_mat_1m: np.ndarray,
                                  fixing_1m: np.ndarray,
                                  n_days_1m: np.ndarray,
                                  prices_1m: np.ndarray):
            return 0.0
        def objective_function_3m(knot_values: np.ndarray,
                                  o_mat_3m: np.ndarray,
                                  fixing_3m: np.ndarray,
                                  n_days_3m: np.ndarray,
                                  prices_3m: np.ndarray):
            return 0.0
        on = o_matrix_1m = stubs_1m = days_1m = px_1m = o_matrix_3m = stubs_3m = days_3m = px_3m = 0

        # If 1m futures are present
        if futures_1m_prices is not None:
            futures_1m = [IRFuture(x) for x in futures_1m_prices.index]
            px_1m = futures_1m_prices.values.squeeze()
            fut_start_end_1m = np.array([convert_date(fut.get_reference_start_end_dates()) for fut in futures_1m])
            days_1m = (np.diff(fut_start_end_1m, axis=1) + 1).squeeze()
            on, stubs_1m = _calculate_stub_fixing(ref_date, fut_start_end_1m, _SOFR_ if "sofr" in self.rate_name.lower() else _FF_, False)
            o_matrix_1m = _create_overlap_matrix(fut_start_end_1m, self._fomc_effective_dates).astype(float)

            def objective_function_1m(knot_values: np.ndarray,
                                      o_mat_1m: np.ndarray,
                                      fixing_1m: np.ndarray,
                                      n_days_1m: np.ndarray,
                                      prices_1m: np.ndarray):
                res_1m = _price_1m_futures(knot_values, o_mat_1m, fixing_1m, n_days_1m)
                return np.sum((res_1m - prices_1m) ** 2)

            self.market_data[f"{self.rate_name.upper()}1M"] = futures_1m_prices


        # If 3m futures are present
        if futures_3m_prices is not None:
            futures_3m = [IRFuture(x) for x in futures_3m_prices.index]
            px_3m = futures_3m_prices.values.squeeze()
            fut_start_end_3m = np.array([convert_date(fut.get_reference_start_end_dates()) for fut in futures_3m])
            days_3m = (np.diff(fut_start_end_3m, axis=1) + 1).squeeze()
            on, stubs_3m = _calculate_stub_fixing(ref_date, fut_start_end_3m, _SOFR_,True)
            o_matrix_3m = _create_overlap_matrix(fut_start_end_3m, self._fomc_effective_dates).astype(float)

            def objective_function_3m(knot_values: np.ndarray,
                                      o_mat_3m: np.ndarray,
                                      fixing_3m: np.ndarray,
                                      n_days_3m: np.ndarray,
                                      prices_3m: np.ndarray):
                res_3m = _price_3m_futures(knot_values, o_mat_3m, fixing_3m, n_days_3m)
                return np.sum((res_3m - prices_3m) ** 2)

            self.market_data[f"{self.rate_name.upper()}3M"] = futures_3m_prices

        def objective_function(knot_values: np.ndarray,
                               o_mat_1m: np.ndarray,
                               fixing_1m: np.ndarray,
                               n_days_1m: np.ndarray,
                               prices_1m: np.ndarray,
                               o_mat_3m: np.ndarray,
                               fixing_3m: np.ndarray,
                               n_days_3m: np.ndarray,
                               prices_3m: np.ndarray):
            loss = objective_function_1m(knot_values, o_mat_1m, fixing_1m, n_days_1m, prices_1m)
            loss += objective_function_3m(knot_values, o_mat_3m, fixing_3m, n_days_3m, prices_3m)
            loss += 1e2 * np.sum(np.diff(knot_values) ** 2)
            loss += fomc * 1e4 * (on - knot_values[0]) ** 2
            return loss

        # Initial values
        initial_knot_values = self._effective_rates
        bounds = Bounds(0.0, 0.08)
        res = minimize(objective_function,
                       initial_knot_values,
                       args=(o_matrix_1m, stubs_1m, days_1m, px_1m,
                             o_matrix_3m, stubs_3m, days_3m, px_3m),
                       method="L-BFGS-B",
                       bounds=bounds)

        # Set curve status
        self._effective_rates = res.x
        return self

    def calculate_sofr_future_swap_spread(self):
        """
        This function uses curve to evaluate future prices as well as equivalent swap rates.
        :return:
        """
        assert self.rate_name == "SOFR"
        fut_3m = self.market_data["SOFR3M"].iloc[1:]    # Remove stub future
        fut_rates = 1e2 - fut_3m.values.squeeze()
        fut_st_et = np.array([convert_date(IRFuture(x).get_reference_start_end_dates()) for x in fut_3m.index])
        fut_st_et[:, 1] += 1    # This is because the end day is one day before IMM, which needs to be accrued.
        forward_rates = 1e2 * self.forward_rates(fut_st_et[:, 0], fut_st_et[:, 1])
        df = pd.Series(fut_rates - forward_rates, index=pd.DatetimeIndex(parse_date(fut_st_et[:, 0])))
        self._future_swap_spread = df
        return self

    def price_1m_futures(self, futures_1m: list[str] | np.ndarray[str]) -> np.ndarray:
        """
        This function prices a list of SOFR 1m futures on a curve
        :param futures_1m:
        :return:
        """
        futures = [IRFuture(x) for x in futures_1m]
        fut_types = [fut.future_type for fut in futures]
        assert len(set(fut_types)) == 1
        fut_type = fut_types[0]
        assert self.rate_name.upper() in fut_type.upper()
        fixing_manager = _SOFR_ if "SOFR" in fut_type.upper() else _FF_

        ref_date = convert_date(self.reference_date)
        st_et = np.array([convert_date(IRFuture(x).get_reference_start_end_dates()) for x in futures_1m])
        o_matrix = _create_overlap_matrix(st_et, self._fomc_effective_dates)
        knot_values = self._effective_rates
        _, stubs = _calculate_stub_fixing(ref_date, st_et, fixing_manager, False)
        n_days = (np.diff(st_et, axis=1) + 1).squeeze()
        return _price_1m_futures(knot_values, o_matrix, stubs, n_days)

    def price_3m_futures(self, futures_3m: list[str] | np.ndarray[str]) -> np.ndarray:
        """
        This function prices a list of SOFR 3m futures on a curve
        :param futures_3m:
        :return:
        """
        futures = [IRFuture(x) for x in futures_3m]
        fut_types = [fut.future_type for fut in futures]
        assert len(set(fut_types)) == 1
        fut_type = fut_types[0]
        assert self.rate_name.upper() in fut_type.upper()

        ref_date = convert_date(self.reference_date)
        st_et = np.array([convert_date(IRFuture(x).get_reference_start_end_dates()) for x in futures_3m])
        o_matrix = _create_overlap_matrix(st_et, self._fomc_effective_dates)
        _, stubs = _calculate_stub_fixing(ref_date, st_et, _SOFR_,True)
        n_days = (np.diff(st_et, axis=1) + 1).squeeze()
        return _price_3m_futures(self._effective_rates, o_matrix, stubs, n_days)

    def price_swap_rates(self, swaps: list[SOFRSwap]) -> np.ndarray:
        """
        This function prices a list of SOFR swaps
        :param swaps:
        :return:
        """
        ref_date = convert_date(self.reference_date)
        schedules = [swap.get_float_leg_schedule(True).values for swap in swaps]
        partition = np.array([len(x) for x in schedules])
        schedule_block = np.concatenate(schedules, axis=0)
        schedules = schedule_block[:, :-1]
        dcfs = schedule_block[:, -1].squeeze()
        return _price_swap_rates(self._swap_knot_values,
                                 ref_date,
                                 self._swap_knot_dates,
                                 schedules,
                                 dcfs,
                                 partition
                                 )

    def price_spot_rates(self, tenors: list[str] | np.ndarray[str]) -> np.ndarray:
        """
        This function prices a list of spot starting par rates given tenor
        :param tenors:
        :return:
        """
        swaps = [SOFRSwap(self.reference_date, tenor=x) for x in tenors]
        return self.price_swap_rates(swaps)

    def shock_swap_curve_with_convexity(self):
        """
        This method shocks the swap curve with minimal amount to preserve the convexity
        :return:
        """
        initial_knot_values = self._swap_knot_values.copy()

        # Get 3M future prices
        fut_3m = self.market_data.get("SOFR3M", pd.Series())

    def shock_future_curve_with_convexity(self):
        pass

########################################################################################################################
# External functionalities
########################################################################################################################
def shock_curve(curve: USDCurve,
                shock_amount: np.ndarray,
                shock_type: str,
                new_curve=False,
                preserve_convexity=True) -> USDCurve:
    """
    This function shocks a USD curve
    :param preserve_convexity: If shocking effective_rate [or zero_rate], additionally shock zero_rate [or effective_rate]
    by a minimal change to preserve convexity
    :param new_curve:
    :param shock_amount:
    :param curve:
    :param shock_type:
    :return:
    """
    assert shock_type in ["parallel_bps", "effective_rate", "zero_rate"]
    output_curve = curve
    if new_curve:
        output_curve = deepcopy(curve)
    if shock_type == "parallel_bps":
        output_curve.shock_zero_rate(shock_amount)
        output_curve.shock_zero_rate(shock_amount)
    elif shock_type == "effective_rate":
        output_curve.set_effective_rate(shock_amount)
        if preserve_convexity:
            assert isinstance(output_curve.convexity_model, float)
            output_curve.shock_swap_curve_with_convexity()
    else:
        output_curve.set_zero_rate(shock_amount)
        if preserve_convexity:
            assert isinstance(output_curve.convexity_model, float)
            output_curve.shock_future_curve_with_convexity()
    return output_curve


########################################################################################################################
# Debug
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
    "SFRU8": 96.525,
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

def debug_ff_calibration():
    # FF
    ff = USDCurve("FF", "2024-10-09")
    ff.calibrate_future_curve(ff_prices)
    print("Pricing errors in bps for FF futures:")
    err = 1e2 * (ff.price_1m_futures(ff_prices.index) - ff_prices.values)
    print(err)
    ff.plot_effective_rates(10, 6)

def debug_sofr1m_calibration():
    # SOFR1M
    sofr1m = USDCurve("SOFR", "2024-10-09")
    sofr1m.calibrate_future_curve(sofr_1m_prices)
    print("Pricing errors in bps for SOFR1M futures:")
    err = 1e2 * (sofr1m.price_1m_futures(sofr_1m_prices.index) - sofr_1m_prices.values)
    print(err)
    sofr1m.plot_effective_rates(10, 6)

def debug_sofr3m_calibration():
    # SOFR3M
    sofr3m = USDCurve("SOFR", "2024-10-09")
    sofr3m.calibrate_future_curve(futures_3m_prices=sofr_3m_prices)
    print("Pricing errors in bps for SOFR3M futures:")
    err = 1e2 * (sofr3m.price_3m_futures(sofr_3m_prices.index) - sofr_3m_prices.values)
    print(err)
    sofr3m.plot_effective_rates(10, 6)

def debug_sofr_calibration():
    # SOFR1M and 3M
    sofr = USDCurve("SOFR", "2024-10-09")
    sofr.calibrate_future_curve(sofr_1m_prices, sofr_3m_prices)
    print("Pricing errors in bps for SOFR1M futures:")
    err = 1e2 * (sofr.price_1m_futures(sofr_1m_prices.index) - sofr_1m_prices.values)
    print(err)
    print("Pricing errors in bps for SOFR3M futures:")
    err3 = 1e2 * (sofr.price_3m_futures(sofr_3m_prices.index) - sofr_3m_prices.values)
    print(err3)
    sofr.plot_effective_rates(10, 6)

def debug_sofr_swap_calibration():
    sofr = USDCurve("SOFR", "2024-10-09")
    sofr.calibrate_swap_curve(sofr_swaps_rates)
    print("Pricing errors in bps for swaps")
    print(1e2 * (sofr.price_spot_rates(sofr_swaps_rates.index) - sofr_swaps_rates.values))
    sofr.plot_swap_zero_rates()

def debug_sofr_swap_calibration_with_convexity():
    sofr = USDCurve("SOFR", "2024-10-09")
    sofr.calibrate_swap_curve_with_convexity(sofr_3m_prices, sofr_swaps_rates)
    print("Pricing errors in bps for swaps")
    print(1e2 * (sofr.price_spot_rates(sofr_swaps_rates.index) - sofr_swaps_rates.values))
    sofr.plot_swap_zero_rates()

    sofr.calculate_sofr_future_swap_spread()
    sofr.plot_sofr_future_swap_spread()


# Example usage
if __name__ == '__main__':
    # debug_sofr_swap_calibration()
    debug_sofr_swap_calibration_with_convexity()
    exit(0)

