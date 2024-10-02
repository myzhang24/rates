import pandas as pd
from pandas.tseries.offsets import MonthEnd
from datetime import datetime
from holiday import SIFMA


class SOFRFuturesBase:
    def __init__(self, ticker):
        self.ticker = ticker.upper()
        self.price = 100.0
        self.holiday_calendar = SIFMA.holiday_set  # SIFMA holidays
        self.parse_ticker()
        self.set_contract_details()

    def load_price(self, price):
        if 2 < price < 150:
            self.price = price
        if 0 < price < 1.5:
            self.price = price * 100
        self.price = 100.0

    def parse_ticker(self):
        # This method will be overridden in child classes
        pass

    def set_contract_details(self):
        # This method will be overridden in child classes
        pass

    def is_business_day(self, date):
        date = pd.Timestamp(date)
        return date.weekday() < 5 and date not in self.holiday_calendar

    def adjust_to_business_day(self, date, convention='following'):
        if convention == 'following':
            while not self.is_business_day(date):
                date += pd.Timedelta(days=1)
        elif convention == 'preceding':
            while not self.is_business_day(date):
                date -= pd.Timedelta(days=1)
        else:
            raise ValueError("Unsupported convention. Use 'following' or 'preceding'.")
        return date

    def get_next_ticker(self):
        # This method will be overridden in child classes
        pass

    def get_previous_ticker(self):
        # This method will be overridden in child classes
        pass


class SOFR1MFutures(SOFRFuturesBase):
    MONTH_CODES = {
        'F': 1,  # January
        'G': 2,  # February
        'H': 3,  # March
        'J': 4,  # April
        'K': 5,  # May
        'M': 6,  # June
        'N': 7,  # July
        'Q': 8,  # August
        'U': 9,  # September
        'V': 10,  # October
        'X': 11,  # November
        'Z': 12  # December
    }

    def __init__(self, ticker):
        super().__init__(ticker)
        self.reference_end_date = None
        self.reference_start_date = None
        self.expiry_date = None
        self.month = None
        self.year = None
        self.year_code = None
        self.month_code = None
        self.ticker = ticker
        self.parse_ticker()
        self.set_contract_details()

    def parse_ticker(self):
        # Ticker format: SER + Month Code + Year Digit(s)
        if not self.ticker.startswith('SER'):
            raise ValueError("Invalid ticker format for SOFR 1M Futures.")
        code = self.ticker[3:]
        self.month_code = code[0]
        self.year_code = code[1:]
        if self.month_code not in self.MONTH_CODES:
            raise ValueError("Invalid month code in ticker.")
        # Handle one or two-digit year codes (e.g., '5' for 2005 or '25' for 2025)
        if len(self.year_code) == 1:
            self.year = 2020 + int(self.year_code)
        elif len(self.year_code) == 2:
            self.year = 2000 + int(self.year_code)
        else:
            raise ValueError("Invalid year code in ticker.")
        self.month = self.MONTH_CODES[self.month_code]

    def set_contract_details(self):
        # Set the expiry date (last business day of the contract month)
        self.expiry_date = self.get_expiry_date()
        # Set the reference period start and end dates
        self.reference_start_date, self.reference_end_date = self.get_reference_period()

    def get_expiry_date(self):
        # SOFR 1M Futures expire on the last business day of the contract month
        last_day = pd.Timestamp(datetime(self.year, self.month, 1)) + MonthEnd(0)
        expiry_date = self.adjust_to_business_day(last_day, convention='preceding')
        return expiry_date

    def get_reference_period(self):
        # Reference period is the contract month itself
        start_date = pd.Timestamp(datetime(self.year, self.month, 1))
        end_date = start_date + MonthEnd(0)
        return start_date, end_date

    def adjust_to_business_day(self, date, convention='preceding'):
        # Adjust the date to the previous business day if it falls on a weekend or holiday
        # For simplicity, we'll assume weekends are non-business days
        while date.weekday() > 4:  # 0 = Monday, 6 = Sunday
            date -= pd.Timedelta(days=1)
        return date

    def get_next_ticker(self):
        # Increment the month and year as needed
        next_month = self.month + 1
        next_year = self.year
        if next_month > 12:
            next_month = 1
            next_year += 1
        # Find the month code
        month_code = [code for code, month in self.MONTH_CODES.items() if month == next_month][0]
        year_code = str(next_year % 100).zfill(2)
        next_ticker = f"SER{month_code}{year_code}"
        return next_ticker

    def get_previous_ticker(self):
        # Decrement the month and year as needed
        prev_month = self.month - 1
        prev_year = self.year
        if prev_month < 1:
            prev_month = 12
            prev_year -= 1
        # Find the month code
        month_code = [code for code, month in self.MONTH_CODES.items() if month == prev_month][0]
        year_code = str(prev_year % 100).zfill(2)
        prev_ticker = f"SER{month_code}{year_code}"
        return prev_ticker


class SOFR3MFutures(SOFRFuturesBase):
    QUARTERLY_MONTHS = {
        'H': 3,  # March
        'M': 6,  # June
        'U': 9,  # September
        'Z': 12  # December
    }

    def __init__(self, ticker):
        super().__init__(ticker)
        self.reference_end_date = None
        self.reference_start_date = None
        self.expiry_date = None
        self.month = None
        self.year = None
        self.year_code = None
        self.month_code = None
        self.ticker = ticker
        self.parse_ticker()
        self.set_contract_details()

    def parse_ticker(self):
        # Ticker format: SFR + Month Code + Year Digit(s)
        if not self.ticker.startswith('SFR'):
            raise ValueError("Invalid ticker format for SOFR 3M Futures.")
        code = self.ticker[3:]
        self.month_code = code[0]
        self.year_code = code[1:]
        if self.month_code not in self.QUARTERLY_MONTHS:
            raise ValueError("Invalid month code in ticker.")
        # Handle one or two-digit year codes
        if len(self.year_code) == 1:
            self.year = 2020 + int(self.year_code)
        elif len(self.year_code) == 2:
            self.year = 2000 + int(self.year_code)
        else:
            raise ValueError("Invalid year code in ticker.")
        self.month = self.QUARTERLY_MONTHS[self.month_code]

    def set_contract_details(self):
        # Set the expiry date (Third Wednesday of the contract month)
        self.expiry_date = self.get_expiry_date()
        # Set the reference period start and end dates
        self.reference_start_date, self.reference_end_date = self.get_reference_period()

    def get_expiry_date(self):
        # SOFR 3M Futures expire on the third Wednesday of the contract month
        expiry_date = self.get_nth_weekday_of_month(self.year, self.month, 3, 2)  # 3rd Wednesday
        return expiry_date

    def get_reference_period(self):
        # Reference period is from the previous IMM date to the day before the contract's expiry date
        start_date = self.get_previous_imm_date()
        end_date = self.expiry_date - pd.Timedelta(days=1)
        return start_date, end_date

    def get_previous_imm_date(self):
        # Calculate the third Wednesday three months before the contract month
        prev_month = self.month - 3
        prev_year = self.year
        if prev_month < 1:
            prev_month += 12
            prev_year -= 1
        prev_imm_date = self.get_nth_weekday_of_month(prev_year, prev_month, 3, 2)  # 3rd Wednesday
        return prev_imm_date

    def get_nth_weekday_of_month(self, year, month, n, weekday):
        # weekday: Monday=0, Sunday=6
        first_day = pd.Timestamp(datetime(year, month, 1))
        days_until_weekday = (weekday - first_day.weekday() + 7) % 7
        nth_weekday = first_day + pd.Timedelta(days=days_until_weekday) + pd.Timedelta(weeks=n - 1)
        return nth_weekday

    def adjust_to_business_day(self, date, convention='following'):
        # Adjust the date to the next business day if it falls on a weekend
        while date.weekday() > 4:  # 0 = Monday, 6 = Sunday
            if convention == 'following':
                date += pd.Timedelta(days=1)
            elif convention == 'preceding':
                date -= pd.Timedelta(days=1)
        return date

    def get_next_ticker(self):
        # Move to the next quarterly month
        months = sorted(self.QUARTERLY_MONTHS.values())
        current_index = months.index(self.month)
        if current_index == len(months) - 1:
            next_month = months[0]
            next_year = self.year + 1
        else:
            next_month = months[current_index + 1]
            next_year = self.year
        # Find the month code
        month_code = [code for code, month in self.QUARTERLY_MONTHS.items() if month == next_month][0]
        year_code = str(next_year % 100).zfill(2)
        next_ticker = f"SFR{month_code}{year_code}"
        return next_ticker

    def get_previous_ticker(self):
        # Move to the previous quarterly month
        months = sorted(self.QUARTERLY_MONTHS.values())
        current_index = months.index(self.month)
        if current_index == 0:
            prev_month = months[-1]
            prev_year = self.year - 1
        else:
            prev_month = months[current_index - 1]
            prev_year = self.year
        # Find the month code
        month_code = [code for code, month in self.QUARTERLY_MONTHS.items() if month == prev_month][0]
        year_code = str(prev_year % 100).zfill(2)
        prev_ticker = f"SFR{month_code}{year_code}"
        return prev_ticker


def get_sofr_1m_futures(reference_date):
    """
    Returns the tickers for the 13 consecutive live SOFR 1M futures (including stub).

    Parameters:
    - reference_date: datetime, the reference date for which the futures are live.

    Returns:
    - list of tickers for 13 consecutive SOFR 1M futures.
    """
    tickers = []
    months = ['F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z']  # CME futures month codes
    year = reference_date.year
    month_index = reference_date.month - 1  # Index for the current month in futures code

    for i in range(13):
        ticker_month = months[month_index % 12]
        ticker_year = str(year % 100)  # Use the last digit of the year for the ticker

        ticker = f'SER{ticker_month}{ticker_year}'  # Assuming 'SER' prefix for SOFR 1M futures
        tickers.append(ticker)

        # Increment the month and adjust the year when needed
        month_index += 1
        if month_index % 12 == 0:
            year += 1  # Increment the year after December

    return tickers


def get_sofr_3m_futures(reference_date):
    """
    Returns the tickers for the 16 consecutive live SOFR 3M futures (including stub).

    Parameters:
    - reference_date: datetime, the reference date for which the futures are live.

    Returns:
    - list of tickers for 16 consecutive SOFR 3M futures.
    """
    tickers = []
    months = ['H', 'M', 'U', 'Z']  # CME quarterly futures month codes for March, June, September, December
    year = reference_date.year
    month_index = (reference_date.month - 1) // 3  # Quarterly month index

    for i in range(16):
        ticker_month = months[month_index % 4]
        ticker_year = str(year % 100)  # Use the last digit of the year for the ticker

        ticker = f'SFR{ticker_month}{ticker_year}'  # Assuming 'SER' prefix for SOFR 3M futures
        tickers.append(ticker)

        # Increment the quarter and adjust the year when needed
        month_index += 1
        if month_index % 4 == 0:
            year += 1  # Increment the year after December quarter

    return tickers


if __name__ == '__main__':
    # Example for SOFR 1M Futures
    sofr1m = SOFR1MFutures('SERM4')
    print("SOFR 1M Futures:")
    print(f"Ticker: {sofr1m.ticker}")
    print(f"Expiry Date: {sofr1m.expiry_date.date()}")
    print(f"Reference Start Date: {sofr1m.reference_start_date.date()}")
    print(f"Reference End Date: {sofr1m.reference_end_date.date()}")
    print(f"Next Ticker: {sofr1m.get_next_ticker()}")
    print(f"Previous Ticker: {sofr1m.get_previous_ticker()}")

    # Example for SOFR 3M Futures
    sofr3m = SOFR3MFutures('SFRU4')
    print("\nSOFR 3M Futures:")
    print(f"Ticker: {sofr3m.ticker}")
    print(f"Expiry Date: {sofr3m.expiry_date.date()}")
    print(f"Reference Start Date: {sofr3m.reference_start_date.date()}")
    print(f"Reference End Date: {sofr3m.reference_end_date.date()}")
    print(f"Next Ticker: {sofr3m.get_next_ticker()}")
    print(f"Previous Ticker: {sofr3m.get_previous_ticker()}")

    # Example of generation
    reference_date = datetime(2023, 3, 15)
    tickers_3m = get_sofr_3m_futures(reference_date)
    tickers_1m = get_sofr_1m_futures(reference_date)
    print(f"\nThe 13 SOFR 1M futures as of {reference_date.date()} are")
    print(tickers_1m)
    print(f"The 16 SOFR 3M futures as of {reference_date.date()} are")
    print(tickers_3m)
