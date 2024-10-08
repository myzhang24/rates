import datetime as dt
import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd

from holiday import SIFMA, NYFED


# Auxiliary functions
def get_nth_weekday_of_month(year, month, n, weekday):
    # weekday: Monday=0, Sunday=6
    first_day = dt.datetime(year, month, 1)
    days_until_weekday = (weekday - first_day.weekday() + 7) % 7
    nth_weekday = first_day + pd.Timedelta(days=days_until_weekday) + pd.Timedelta(weeks=n - 1)
    return nth_weekday


# SOFR Future base class
class SOFRFuturesBase:
    def __init__(self, ticker):
        self.ticker = ticker.upper()
        self.month = None
        self.year = None
        self.reference_start_date = None
        self.reference_end_date = None
        self.expiry_date = None
        self.parse_ticker()

    def parse_ticker(self):
        # This method will be overridden in child classes
        pass

    def get_next_ticker(self):
        # This method will be overridden in child classes
        pass

    def get_previous_ticker(self):
        # This method will be overridden in child classes
        pass

    def reference_array(self):
        # This method will be overridden in child classes
        pass


# SOFR 1M futures class
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
        self.parse_ticker()

    def parse_ticker(self):
        # Ticker format: SER + Month Code + Year Digit(s)
        if not self.ticker.startswith('SER'):
            raise ValueError("Invalid ticker format for SOFR 1M Futures.")
        code = self.ticker[3:]
        month_code = code[0]
        year_code = code[1:]
        if month_code not in self.MONTH_CODES:
            raise ValueError("Invalid month code in ticker.")
        # Handle one or two-digit year codes (e.g., '5' for 2005 or '25' for 2025)
        if len(year_code) == 1:
            self.year = 2020 + int(year_code)
        elif len(year_code) == 2:
            self.year = 2000 + int(year_code)
        else:
            raise ValueError("Invalid year code in ticker.")
        self.month = self.MONTH_CODES[month_code]

        # 1M futures reference to first to last calendar day
        start_date = dt.datetime(self.year, self.month, 1)
        end_date = (start_date + MonthEnd(0)).to_pydatetime()
        self.expiry_date = SIFMA.prev_biz_day(end_date, 0)
        self.reference_start_date = start_date
        self.reference_end_date = end_date

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

    def reference_array(self):
        """
        Returns a reference array of SOFR reference days, first of month to end of month calendar days inclusive
        :return:
        """
        dates = pd.date_range(self.reference_start_date, self.reference_end_date, freq="1D")
        return dates

class SOFR3MFutures(SOFRFuturesBase):
    QUARTERLY_MONTHS = {
        'H': 3,  # March
        'M': 6,  # June
        'U': 9,  # September
        'Z': 12  # December
    }

    def __init__(self, ticker):
        super().__init__(ticker)
        self.parse_ticker()

    def parse_ticker(self):
        # Ticker format: SFR + Month Code + Year Digit(s)
        if not self.ticker.startswith('SFR'):
            raise ValueError("Invalid ticker format for SOFR 3M Futures.")
        code = self.ticker[3:]
        month_code = code[0]
        year_code = code[1:]
        if month_code not in self.QUARTERLY_MONTHS:
            raise ValueError("Invalid month code in ticker.")
        # Handle one or two-digit year codes
        if len(year_code) == 1:
            self.year = 2020 + int(year_code)
        elif len(year_code) == 2:
            self.year = 2000 + int(year_code)
        else:
            raise ValueError("Invalid year code in ticker.")
        self.month = self.QUARTERLY_MONTHS[month_code]

        # SOFR 3M Futures expire on the third Wednesday of the contract month


        # Reference period is from the previous IMM date to the day before the contract's expiry date
        self.reference_start_date = get_nth_weekday_of_month(self.year, self.month, 3, 2)  # 3rd Wednesday
        self.expiry_date = self.get_next_imm_date()
        self.reference_end_date = self.expiry_date - dt.timedelta(days=1)

    def get_next_imm_date(self):
        # Calculate the third Wednesday three months before the contract month
        next_month = self.month + 3
        next_year = self.year
        if next_month > 12:
            next_month -= 12
            next_year += 1
        next_imm_date = get_nth_weekday_of_month(next_year, next_month, 3, 2)  # 3rd Wednesday
        return next_imm_date

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

    def reference_array(self):
        dates = NYFED.biz_date_range(self.reference_start_date, self.reference_end_date)
        if self.reference_start_date not in dates:
            dates = np.insert(dates, 0 , self.reference_start_date)
        if self.reference_end_date not in dates:
            dates = np.insert(dates, -1, self.reference_end_date)
        return dates

if __name__ == '__main__':
    # Example for SOFR 1M Futures
    sofr1m = SOFR1MFutures('SERM4')
    print("SOFR 1M Futures:")
    print(f"Ticker: {sofr1m.ticker}")
    print(f"Expiry Date: {sofr1m.expiry_date}")
    print(f"Reference Start Date: {sofr1m.reference_start_date}")
    print(f"Reference End Date: {sofr1m.reference_end_date}")
    print(f"Next Ticker: {sofr1m.get_next_ticker()}")
    print(f"Previous Ticker: {sofr1m.get_previous_ticker()}")
    sofr1m.reference_array()

    # Example for SOFR 3M Futures
    sofr3m = SOFR3MFutures('SFRU4')
    print("\nSOFR 3M Futures:")
    print(f"Ticker: {sofr3m.ticker}")
    print(f"Expiry Date: {sofr3m.expiry_date}")
    print(f"Reference Start Date: {sofr3m.reference_start_date}")
    print(f"Reference End Date: {sofr3m.reference_end_date}")
    print(f"Next Ticker: {sofr3m.get_next_ticker()}")
    print(f"Previous Ticker: {sofr3m.get_previous_ticker()}")
    sofr3m.reference_array()
