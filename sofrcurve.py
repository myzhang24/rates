import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from holiday import SIFMA, NYFED
from swaps import SOFRSwap, SOFRFRA, fra_start_end_date
from futures import SOFR1MFutures, SOFR3MFutures
from fomc import generate_fomc_meeting_dates
from fixings import load_fixings


def last_published_value(reference_dates, knot_dates, knot_values):
    indices = np.searchsorted(knot_dates, reference_dates, side='right') - 1

    # Initialize D with NaN values
    res = np.full(len(reference_dates), np.nan)

    # Assign values from C to D where valid indices are found
    valid_mask = indices >= 0
    res[valid_mask] = knot_values[indices[valid_mask]]
    return res


def price_1m_future(reference_dates, knot_dates, knot_values, fixings=None):
    if reference_dates[0] < knot_dates[0]:
        fixings_to_use = fixings[reference_dates[0]:knot_dates[0]]
        knots = np.concatenate(fixings_to_use.index.to_numpy(), knot_dates)
        values = np.concatenate(fixings_to_use.values, knot_values)
    else:
        knots = knot_dates
        values = knot_values
    reference_rates = last_published_value(reference_dates, knots, values)
    return 1e2 * (1 - reference_rates.mean())


def price_3m_future(reference_dates, knot_dates, knot_values, fixings=None):
    if reference_dates[0] < knot_dates[0]:
        fixings_to_use = fixings[reference_dates[0]:knot_dates[0]]
        knots = np.concatenate(fixings_to_use.index.to_numpy(), knot_dates)
        values = np.concatenate(fixings_to_use.values, knot_values)
    else:
        knots = knot_dates
        values = knot_values
    reference_rates = last_published_value(reference_dates, knots, values)
    return 1e2 * (1 - sofr_compound(reference_dates, reference_rates))


def sofr_compound(reference_dates, reference_rates):
    annualized_rate = np.prod(1 + reference_rates * reference_dates.diff().days() / 360) - 1
    return 360 * annualized_rate / (reference_dates[-1] - reference_dates[0]).days


class USDSOFRCurve:
    def __init__(self, reference_date):
        self.reference_date = pd.Timestamp(reference_date)

        self.market_instruments = []
        self.sofr_1m_futures = []
        self.sofr_3m_futures = []
        self.sofr_fras = []
        self.sofr_swaps = []

        self.future_knot_dates = None
        self.future_knot_values = None
        self.initialize_future_knot_dates()

        self.swap_knot_dates = None
        self.swap_knot_values = None

        self.fixings = load_fixings(self.reference_date - pd.DateOffset(months=4), self.reference_date)

    def initialize_future_knot_dates(self):
        # Initialize future knots
        meeting_dates = generate_fomc_meeting_dates(self.reference_date.date(),
                                                    (self.reference_date + pd.DateOffset(years=2)).date())
        effective_dates = [SIFMA.next_biz_day(x, 1) for x in meeting_dates]
        next_biz_day = SIFMA.next_biz_day(self.reference_date, 0)
        self.future_knot_dates = np.array(effective_dates)
        if next_biz_day not in effective_dates:
            self.future_knot_dates = np.array([next_biz_day] + effective_dates)
        self.future_knot_values = 0.03 * np.ones((1, len(self.future_knot_dates)))

    def initialize_swap_knot_dates(self):
        # Initialize knots for swaps
        next_biz_day = SIFMA.next_biz_day(self.reference_date, 0)
        swap_dates = [x.fixed_leg_schedule[-1]["Payment Date"] for x in self.sofr_swaps]
        self.swap_knot_dates = np.array([next_biz_day] + swap_dates)
        self.swap_knot_values = 0.03 * np.ones((1, len(self.swap_knot_dates)))

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

    def futures_objective_function(self, knot_values):
        """
        Build the constant meeting daily forward futures curve
        :return:
        """
        res = 0.0

        for fut in self.sofr_1m_futures:
            reference_dates = pd.date_range(fut.reference_start_date, fut.reference_end_date)
            if fut.reference_start_date < self.reference_date:
                price = price_1m_future(reference_dates, self.future_knot_dates, knot_values)
            else:
                price = price_1m_future(reference_dates, self.future_knot_dates, knot_values)
            res += 0.5 * (fut.price - price) ** 2

        for fut in self.sofr_3m_futures:
            reference_dates = NYFED.biz_date_range(fut.reference_start_date, fut.reference_end_date)
            price = price_3m_future(reference_dates, self.future_knot_dates, self.future_knot_values)
            res += (fut.price - price) ** 2

        res += 1e2 * np.sum(np.diff(knot_values) ** 2)
        return res

    def build_future_curve(self):
        initial_rates = self.future_knot_values
        bounds = (0.0, 0.20)
        result = least_squares(self.futures_objective_function, initial_rates, bounds=bounds)
        self.future_knot_values = result.x
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
    exit(0)
