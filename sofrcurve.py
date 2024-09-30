import numpy as np
from scipy.optimize import least_squares
from datetime import datetime, timedelta
from holiday import SIFMA, NYFED
from scheduler import SOFRSwapScheduler
from futures import SOFR1MFutures, SOFR3MFutures
from fomc import generate_fomc_meeting_dates


class UsdSofrCurveBuilder:
    def __init__(self, reference_date):
        self.reference_date = reference_date
        self.sifma_calendar = SIFMA_Calendar()
        self.nyfed_calendar = NYFED_Calendar()
        self.schedule_generator = ScheduleGenerator()
        self.futures_utility = FuturesUtility()
        self.fomc_dates = FOMC_Dates().get_future_meetings(reference_date)
        self.market_instruments = []
        self.curve_nodes = []
        self.discounts = {}
        self.forward_rates = {}

    def load_market_data(self, sofr_1m_futures, sofr_3m_futures, sofr_fras, sofr_ois_swaps):
        """
        Load market data for calibration.

        Parameters:
        - sofr_1m_futures: list of tuples (ticker, price)
        - sofr_3m_futures: list of tuples (ticker, price)
        - sofr_fras: list of tuples (start_date, end_date, rate)
        - sofr_ois_swaps: list of tuples (tenor, rate)
        """
        self.market_instruments = {
            '1M_Futures': sofr_1m_futures,
            '3M_Futures': sofr_3m_futures,
            'FRAs': sofr_fras,
            'OIS_Swaps': sofr_ois_swaps
        }

    def convexity_adjustment(self, start_date):
        """
        Calculate the convexity adjustment using a quadratic model.

        Parameters:
        - start_date: datetime

        Returns:
        - adjustment: float
        """
        time = (start_date - self.reference_date).days / 365.25
        # Quadratic model: adjustment = a * time^2
        a = 0.0001  # Example coefficient, should be calibrated or given
        adjustment = a * time ** 2
        return adjustment

    def build_curve(self):
        """
        Build the USD SOFR curve by calibrating to market instruments.
        """
        # Initial guess for the optimization
        initial_rates = [0.02] * (len(self.market_instruments['1M_Futures']) +
                                  len(self.market_instruments['3M_Futures']) +
                                  len(self.market_instruments['FRAs']) +
                                  len(self.market_instruments['OIS_Swaps']))
        # Bounds for the rates
        bounds = (0.0, 1.0)

        # Optimize to minimize the difference between market and model prices
        result = least_squares(self.objective_function, initial_rates, bounds=bounds)

        # Extract the calibrated zero rates
        self.curve_nodes = result.x

        # Build discount factors and forward rates
        self.construct_discount_curve()

    def objective_function(self, rates):
        """
        Objective function for optimization.

        Parameters:
        - rates: list of zero rates

        Returns:
        - residuals: list of differences between market and model prices
        """
        residuals = []

        idx = 0
        # 1M SOFR Futures
        for ticker, market_price in self.market_instruments['1M_Futures']:
            start_date, end_date = self.futures_utility.get_future_dates(ticker)
            model_price = self.price_sofr_future(start_date, end_date, rates[idx])
            residuals.append(model_price - market_price)
            idx += 1

        # 3M SOFR Futures
        for ticker, market_price in self.market_instruments['3M_Futures']:
            start_date, end_date = self.futures_utility.get_future_dates(ticker)
            adjustment = self.convexity_adjustment(start_date)
            model_price = self.price_sofr_future(start_date, end_date, rates[idx], adjustment)
            residuals.append(model_price - market_price)
            idx += 1

        # SOFR FRAs
        for start_date, end_date, market_rate in self.market_instruments['FRAs']:
            adjustment = self.convexity_adjustment(start_date)
            model_rate = self.forward_rate(start_date, end_date, rates[idx], adjustment)
            residuals.append(model_rate - market_rate)
            idx += 1

        # SOFR OIS Swaps
        for tenor, market_rate in self.market_instruments['OIS_Swaps']:
            model_rate = self.price_ois_swap(tenor, rates[idx])
            residuals.append(model_rate - market_rate)
            idx += 1

        return residuals

    def construct_discount_curve(self):
        """
        Construct discount factors and forward rates from calibrated zero rates.
        """
        dates = [self.reference_date]
        discounts = [1.0]

        for idx, rate in enumerate(self.curve_nodes):
            # Assuming each rate corresponds to a time interval
            time = (idx + 1) / 12  # Assuming monthly intervals
            date = self.reference_date + timedelta(days=365.25 * time)
            df = np.exp(-rate * time)
            discounts.append(df)
            dates.append(date)

        self.discounts = dict(zip(dates, discounts))

        # Calculate forward rates between dates
        self.forward_rates = {}
        for i in range(len(dates) - 1):
            dt = (dates[i + 1] - dates[i]).days / 365.25
            df1 = discounts[i]
            df2 = discounts[i + 1]
            fwd_rate = (df1 / df2 - 1) / dt
            self.forward_rates[dates[i]] = fwd_rate

        # Apply piecewise constant forward rates between FOMC dates
        self.apply_fomc_constraints()

    def apply_fomc_constraints(self):
        """
        Ensure forward rates are constant between FOMC rate effective dates for the next 2 years.
        """
        fomc_effective_dates = [date + timedelta(days=1) for date in self.fomc_dates[:6]]
        for i in range(len(fomc_effective_dates) - 1):
            start_date = fomc_effective_dates[i]
            end_date = fomc_effective_dates[i + 1]
            rate = self.forward_rates.get(start_date, None)
            if rate:
                # Set forward rates to be constant between FOMC dates
                for date in self.forward_rates:
                    if start_date <= date < end_date:
                        self.forward_rates[date] = rate

    def price_sofr_future(self, start_date, end_date, rate, adjustment=0.0):
        """
        Price a SOFR future contract.

        Parameters:
        - start_date: datetime
        - end_date: datetime
        - rate: float
        - adjustment: float

        Returns:
        - price: float
        """
        dt = (end_date - start_date).days / 365.25
        forward_rate = rate + adjustment
        price = 1 - forward_rate * dt
        return price

    def forward_rate(self, start_date, end_date, rate, adjustment=0.0):
        """
        Calculate the forward rate between two dates.

        Parameters:
        - start_date: datetime
        - end_date: datetime
        - rate: float
        - adjustment: float

        Returns:
        - forward_rate: float
        """
        dt = (end_date - start_date).days / 365.25
        forward_rate = rate + adjustment
        return forward_rate

    def price_ois_swap(self, tenor, rate):
        """
        Price a SOFR OIS swap.

        Parameters:
        - tenor: float (in years)
        - rate: float

        Returns:
        - swap_rate: float
        """
        # Generate payment dates
        payment_dates = self.schedule_generator.generate_schedule(
            self.reference_date, tenor, frequency='Annual', calendar=self.sifma_calendar)

        # Calculate the fixed leg
        fixed_leg = sum(
            [self.discounts[date] * rate * self.schedule_generator.day_count_fraction(date) for date in payment_dates])

        # Calculate the floating leg (assumed to be par at initiation)
        floating_leg = 1 - self.discounts[payment_dates[-1]]

        swap_rate = floating_leg / fixed_leg
        return swap_rate

    def get_discount_factor(self, date):
        """
        Get the discount factor for a given date.

        Parameters:
        - date: datetime

        Returns:
        - discount_factor: float
        """
        return self.discounts.get(date, None)

    def get_forward_rate(self, date):
        """
        Get the forward rate for a given date.

        Parameters:
        - date: datetime

        Returns:
        - forward_rate: float
        """
        return self.forward_rates.get(date, None)


# Example usage
if __name__ == '__main__':
    reference_date = datetime(2023, 10, 1)
    curve_builder = UsdSofrCurveBuilder(reference_date)

    # Load market data (dummy data for illustration)
    sofr_1m_futures = [('FUT1', 0.9975)] * 13
    sofr_3m_futures = [('FUT3M1', 0.9925)] * 16
    sofr_fras = [
        (reference_date + timedelta(days=30 * i), reference_date + timedelta(days=30 * (i + 1)), 0.02 + 0.0001 * i) for
        i in range(10)]
    sofr_ois_swaps = [(0.5, 0.02), (1, 0.022), (2, 0.025), (3, 0.0275), (5, 0.03), (7, 0.032), (10, 0.033), (15, 0.035),
                      (20, 0.036), (30, 0.037)]

    curve_builder.load_market_data(sofr_1m_futures, sofr_3m_futures, sofr_fras, sofr_ois_swaps)
    curve_builder.build_curve()

    # Get discount factor and forward rate for a specific date
    date = reference_date + timedelta(days=365)
    df = curve_builder.get_discount_factor(date)
    fwd_rate = curve_builder.get_forward_rate(date)
    print(f"Discount Factor on {date.date()}: {df}")
    print(f"Forward Rate on {date.date()}: {fwd_rate}")
