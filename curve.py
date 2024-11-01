import datetime as dt
from copy import deepcopy
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import minimize, Bounds

from fixing import _SOFR_, _FF_, past_discount
from date_util import (
    _SIFMA_,
    convert_date,
    parse_date,
    generate_fomc_meeting_dates,
)
from math_util import (
    _df,
    _last_published_value,
    _prepare_swap_batch_price,
    _price_swap_rates,
    _calculate_stub_fixing,
    _create_overlap_matrix,
    _price_1m_futures,
    _price_3m_futures,
)
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
        self.future_swap_spread = None

        self._fomc_effective_dates = None
        self._effective_rates = None
        self._swap_knot_dates = None
        self._swap_knot_values = None

        self.initialize_future_knots(self.reference_date + relativedelta(years=4))

    def get_effective_rates(self):
        return pd.Series(np.round(1e2 * self._effective_rates, 4),
                         index=pd.DatetimeIndex(parse_date(self._fomc_effective_dates)),
                         name=self.rate_name)

    def get_zero_rates(self):
        return pd.Series(np.round(1e2 * self._swap_knot_values, 4),
                         index=pd.DatetimeIndex(parse_date(self._swap_knot_dates)),
                         name=self.rate_name)

    def make_swap(self, tenor: str) -> SOFRSwap:
        """
        This function makes a swap from a string. The string can be a spot starting tenor string,
        or it can be a future ticker to create an equivalent swap
        :param tenor:
        :return:
        """
        try:
            fut = IRFuture(tenor)
            return SOFRSwap(self.reference_date,
                            start_date=fut.reference_start_date,
                            maturity_date=fut.reference_end_date + dt.timedelta(days=1))
        except:
            return SOFRSwap(self.reference_date, tenor=tenor)


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
        swap_tenors = self.market_data.get(f"{self.rate_name}Swaps",
                                           pd.Series(index=["1M", "3M", "6M", "9M", "12M",
                                                            "18M", "2Y", "3Y", "4Y", "5Y",
                                                            "7Y", "10Y", "12Y", "15Y", "20Y",
                                                            "25Y", "30Y"])).index
        swap_rates = np.round(self.price_spot_rates(swap_tenors), 4)
        self.market_data[f"{self.rate_name}Swaps"] = pd.Series(swap_rates, index=swap_tenors)
        return self

    def shock_effective_rate(self, value, shock_type: str, reprice=True):
        """
        This method shocks the effective rate
        :param value:
        :param shock_type:
        :param reprice:
        :return:
        """
        if shock_type.lower() == "additive_bps":
            self._effective_rates += value * 1e-4
        elif shock_type.lower() == "replace_%":
            self._effective_rates = value * 1e-2
        else:
            raise Exception("shock_type can only be additive_bps or replace_%")
        if reprice:
            self.reprice_futures()
        return self

    def shock_zero_rate(self, value, shock_type: str, reprice=True):
        """
        This method shocks the swap zero rate
        :param value:
        :param shock_type:
        :param reprice:
        :return:
        """
        if shock_type.lower() == "additive_bps":
            self._swap_knot_values += value * 1e-4
        elif shock_type.lower() == "replace_%":
            self._swap_knot_values = value * 1e-2
        else:
            raise Exception("shock_type can only be additive_bps or replace_%")
        if reprice:
            self.reprice_swaps()
        return self

    def plot_effective_rates(self, n_meetings=8, n_cuts=3):
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
        dfs = _df(convert_date(self.reference_date), dates, self._swap_knot_dates, self._swap_knot_values)
        ref_date = convert_date(self.reference_date)
        # we along the very first date to be in the past to indicate an aged swap
        if dates[0] < ref_date:
            dfs[0] = past_discount(dates[0], ref_date, self.rate_name)
        return dfs

    def forward_rates(self, st: np.ndarray, et: np.ndarray) -> np.ndarray:
        """
        Returns the annualized forward rates from st to et (not accruing over et to et+1)
        :param st:
        :param et:
        :return:
        """
        return 360.0 * (self.swap_discount_factor(st) / self.swap_discount_factor(et) - 1) / (et - st)

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
        df = self.future_swap_spread
        dt_ind = pd.DatetimeIndex([IRFuture(x).reference_start_date for x in df.index])
        plt.plot(dt_ind, 1e2 * df.values)
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
            last_meeting_date = self.reference_date + relativedelta(years=4)

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
        if next_biz_day not in knot_dates:
            knot_dates = np.array([next_biz_day] + knot_dates)
        self._swap_knot_dates = convert_date(knot_dates)
        self._swap_knot_values = 0.05 * np.ones((len(knot_dates),))
        return self

    # @time_it
    def calibrate_swap_curve(self,
                             spot_rates: pd.Series,
                             convexity: pd.Series | np.ndarray = None,
                             on_penalty: bool = True):
        """
        This function calibrates a swap curve's zero rate knots to prices swaps according to an input market
        :param on_penalty:
        :param convexity:
        :param spot_rates:
        :return:
        """
        # Initialize swap knots
        ref_date = convert_date(self.reference_date)
        spot_swaps = [self.make_swap(x) for x in spot_rates.index]
        mkt_rates = spot_rates.values
        swaps = spot_swaps

        # Load overnight rate
        fomc = 0 if not on_penalty else 1e-2 if self.is_fomc else 1  # adjustment of penalization of overnight jumps in optimization
        on = 1e-2 * _SOFR_.get_fixings_asof(self.reference_date, self.reference_date)
        on = 360 * np.log(1 + on / 360) # convert overnight rate to zero rate

        # Dummy initiation if no convexity
        futs = pd.Series()
        fut_rates = np.array([])
        n_fut = 0

        # If convexity, then price futures
        if convexity is not None:
            # Normalize convexity input
            conv = convexity.ravel() if isinstance(convexity, np.ndarray) else convexity.values

            # Reprice futures
            self.reprice_futures()
            fut_name = "SOFR3M" if self.rate_name == "SOFR" else "FF"
            futs = self.market_data[fut_name]
            n_fut = len(futs)
            fut_swaps = [self.make_swap(x) for x in futs.index]
            swaps = fut_swaps + spot_swaps

            # set fra target rate
            fut_rates = 1e2 - futs.values
            assert conv.shape == fut_rates.shape
            fra_rates = fut_rates -  conv
            mkt_rates = np.concatenate([fra_rates, mkt_rates])

        # Make sure there is enough knots
        self.initialize_swap_knots(swaps)

        # Now we generate and merge the schedule array into a huge one with partition recorded.
        schedules, dcfs, partition = _prepare_swap_batch_price(swaps)
        knot_dates = self._swap_knot_dates
        initial_values = 0.05 * np.ones_like(knot_dates)


        def loss_function(knot_values: np.array) -> float:
            rates = _price_swap_rates(knot_values, ref_date, knot_dates, schedules, dcfs, partition)
            loss = np.sum((rates - mkt_rates) ** 2)
            loss += 1e2 * np.sum(np.diff(knot_values) ** 2)
            loss += fomc * 1e4 * (on - knot_values[0]) ** 2
            return loss

        bounds = Bounds(0.0, 0.08)
        res = minimize(loss_function,
                       initial_values,
                       method="L-BFGS-B" if convexity is None else "SLSQP",
                       bounds=bounds)
        self._swap_knot_values = res.x

        # Set curve status
        self.market_data["SOFRSwaps"] = spot_rates
        if convexity is not None:
            fra = _price_swap_rates(self._swap_knot_values, ref_date, knot_dates, schedules, dcfs, partition)[:n_fut]
            self.future_swap_spread = pd.Series(fut_rates - fra, index=futs.index)
        return self

    def calibrate_future_curve(self,
                               futures_1m_prices: pd.Series=None,
                               futures_3m_prices: pd.Series=None,
                               on_penalty: bool=True):
        """
        This function calibrates the sofr futures curve to the 1m and 3m futures prices
        :param on_penalty:
        :param futures_1m_prices:
        :param futures_3m_prices:
        :return:
        """
        # Create the futures
        ref_date = convert_date(self.reference_date)
        fomc = 0 if not on_penalty else 1e-2 if self.is_fomc else 1.0

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
            px_1m = futures_1m_prices.values
            fut_start_end_1m = np.array([convert_date(fut.get_reference_start_end_dates()) for fut in futures_1m])
            days_1m = (np.diff(fut_start_end_1m, axis=1) + 1).ravel()
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
            px_3m = futures_3m_prices.values
            fut_start_end_3m = np.array([convert_date(fut.get_reference_start_end_dates()) for fut in futures_3m])
            days_3m = (np.diff(fut_start_end_3m, axis=1) + 1).ravel()
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
                       method="SLSQP",
                       bounds=bounds)

        # Set curve status
        self._effective_rates = res.x
        return self

    def calculate_future_swap_spread(self, reprice=True):
        """
        This function uses curve to evaluate future prices as well as equivalent swap rates.
        :return:
        """
        rate_name = "SOFR3M" if self.rate_name == "SOFR" else "FF"
        if reprice:
            self.reprice_futures()
        fut = self.market_data[rate_name]

        fut_rates = 1e2 - fut.values
        fra = [self.make_swap(x) for x in fut.index]
        fra_rates = self.price_swap_rates(fra)
        df = pd.Series(fut_rates - fra_rates, index=fut.index)
        self.future_swap_spread = df
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
        n_days = (np.diff(st_et, axis=1) + 1).ravel()
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
        n_days = (np.diff(st_et, axis=1) + 1).ravel()
        return _price_3m_futures(self._effective_rates, o_matrix, stubs, n_days)

    def price_swap_rates(self, swaps: list[SOFRSwap]) -> np.ndarray:
        """
        This function prices a list of SOFR swaps
        :param swaps:
        :return:
        """
        ref_date = convert_date(self.reference_date)
        schedules, dcfs, partition = _prepare_swap_batch_price(swaps)
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
        swaps = [self.make_swap(x) for x in tenors]
        return self.price_swap_rates(swaps)

    def shock_swap_curve_with_convexity(self):
        """
        This method shocks the swap curve with minimal amount to preserve the convexity.
        This assumes that convexity has been computed via calculate_sofr_future_swap_spread
        :return:
        """
        knot_dates = self._swap_knot_dates
        knot_values = self._swap_knot_values
        ref_date = convert_date(self.reference_date)

        # Make sure convexity is calculated
        fut_name = "SOFR3M" if self.rate_name == "SOFR" else "FF"
        futs = self.market_data[fut_name]
        if self.future_swap_spread is None:
            self.calculate_future_swap_spread()
        if not np.all(futs.index == self.future_swap_spread.index):
            self.calculate_future_swap_spread()

        # Get 3M future prices
        fut_rates = 1e2 - futs.values
        fra = [self.make_swap(x) for x in futs.index]
        schedules, dcfs, partition = _prepare_swap_batch_price(fra)
        fra_rates = fut_rates - self.future_swap_spread.values

        # Find the index until which swap zero rates needs to be shocked to match convexity
        initial_value = knot_values.copy()
        def objective_function(ansatz: np.ndarray):
            swap_rates = _price_swap_rates(ansatz, ref_date, knot_dates, schedules, dcfs, partition)
            err = np.sum((swap_rates - fra_rates) ** 2)
            err += 1e2 * np.sum(np.diff(ansatz - knot_values) ** 2)
            return err

        # Initial values
        bounds = Bounds(0.0, 0.08)
        res = minimize(objective_function,
                       initial_value,
                       method="SLSQP",
                       bounds=bounds)

        # Set curve status
        self._swap_knot_values = res.x
        fra_rates = _price_swap_rates(res.x, ref_date, knot_dates, schedules, dcfs, partition)
        self.future_swap_spread = pd.Series(fut_rates - fra_rates, index=futs.index)
        return self

    def shock_future_curve_with_convexity(self):
        """
        This function re-calibrates the future curve using 3m futures priced using swap curve and convexity
        :return:
        """
        # Make sure convexity is calculated
        fut_name = "SOFR3M" if self.rate_name == "SOFR" else "FF"
        futs = live_futures(self.reference_date, fut_name)
        if self.future_swap_spread is None:
            self.calculate_future_swap_spread()
        if self.future_swap_spread.index.to_list() != futs:
            self.calculate_future_swap_spread()

        # Make FRAs
        fras = [self.make_swap(x) for x in futs]
        fra_rates = self.price_swap_rates(fras)
        fut_rates = fra_rates + self.future_swap_spread.values

        # Recalibrate future curve
        fut_pxs = pd.Series(1e2 - fut_rates, index=futs)
        if fut_name == "SOFR3M":
            self.calibrate_future_curve(futures_3m_prices=fut_pxs, on_penalty=False)
        else:
            self.calibrate_future_curve(futures_1m_prices=fut_pxs, on_penalty=False)

        self.calculate_future_swap_spread()
        return self


########################################################################################################################
# External functionalities
########################################################################################################################
def adjust_sofr1m(ff: pd.Series, sofr1m: pd.Series) -> pd.Series:
    """
    This function assumes linear basis with outliers for sofr-ff, and adjust SOFR1m futures prices
    based on FF1m prices + model basis.
    This is done because FF futures has sufficient liquidity and volume, where SOFR1M futures barely trades after front
    contracts, and pricing is often off near the tail.
    :param ff:
    :param sofr1m:
    :return:
    """

    market_basis = ff.values - sofr1m.values
    adjusted_basis = market_basis.copy()

    for i in range(len(market_basis)):
        leave_one_out_values = np.delete(market_basis, i)
        std = np.std(leave_one_out_values)
        threshold = 2 * std
        mean = np.mean(leave_one_out_values)

        if market_basis[i] > mean + threshold:
            adjusted_basis[i] = mean
        elif market_basis[i] < mean - threshold:
            adjusted_basis[i] = mean

    return pd.Series(np.round(ff.values - adjusted_basis, 4), index=sofr1m.index, name="SOFR1M")

def shock_curve(curve: USDCurve,
                shock_target: str,
                shock_amount: float | np.ndarray | tuple[np.ndarray, np.ndarray],
                shock_type: str,
                preserve_convexity=True,
                new_curve=True,
                ) -> USDCurve:
    """
    This function shocks a USD curve
    :param shock_target:
    :param preserve_convexity: If shocking effective_rate [or zero_rate], additionally shock zero_rate [or effective_rate]
    by a minimal change to preserve convexity
    :param new_curve:
    :param shock_amount:
    :param curve:
    :param shock_type:
    :return:
    """
    output_curve = curve
    if preserve_convexity:
        if output_curve.future_swap_spread is None:
            output_curve.calculate_future_swap_spread(True)
        elif output_curve.future_swap_spread.empty:
            output_curve.calculate_future_swap_spread(True)
    if new_curve:
        output_curve = deepcopy(curve)
    if shock_target.lower() == "zero_rate":
        output_curve.shock_zero_rate(shock_amount, shock_type)
        if preserve_convexity:
            output_curve.shock_future_curve_with_convexity()
    elif shock_target.lower() == "effective_rate":
        output_curve.shock_effective_rate(shock_amount, shock_type)
        if preserve_convexity:
            output_curve.shock_swap_curve_with_convexity()
    elif shock_target == "both":
        if isinstance(shock_amount, tuple):
            shock_amount_1, shock_amount_2 = shock_amount
        else:
            shock_amount_1 = shock_amount_2 = shock_amount
        output_curve.shock_effective_rate(shock_amount_1, shock_type)
        output_curve.shock_zero_rate(shock_amount_2, shock_type)
    return output_curve


if __name__ == '__main__':
    pass
