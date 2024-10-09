import datetime as dt
import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd

from holiday import _SIFMA_
from utils import convert_dates, get_nth_weekday_of_month, next_imm_date


# SOFR 1M futures class
class SOFR1MFuture:
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
        self.ticker = ticker.upper()
        self.reference_start_date = None
        self.reference_end_date = None
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
            year = 2020 + int(year_code)
        elif len(year_code) == 2:
            year = 2000 + int(year_code)
        else:
            raise ValueError("Invalid year code in ticker.")
        month = self.MONTH_CODES[month_code]

        # 1M futures reference to first to last calendar day
        self.reference_start_date = dt.datetime(year, month, 1)
        self.reference_end_date = (self.reference_start_date + MonthEnd(0)).to_pydatetime()

    def reference_array(self):
        """
        Returns a reference array of SOFR reference days, first of month to end of month calendar days inclusive
        :return:
        """
        return convert_dates(pd.date_range(self.reference_start_date, self.reference_end_date, freq="1D"))


class SOFR3MFuture:
    QUARTERLY_MONTHS = {
        'H': 3,  # March
        'M': 6,  # June
        'U': 9,  # September
        'Z': 12  # December
    }

    def __init__(self, ticker):
        self.ticker = ticker.upper()
        self.reference_start_date = None
        self.reference_end_date = None
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
            year = 2020 + int(year_code)
        elif len(year_code) == 2:
            year = 2000 + int(year_code)
        else:
            raise ValueError("Invalid year code in ticker.")
        month = self.QUARTERLY_MONTHS[month_code]

        # SOFR 3M Futures expire on the third Wednesday of the contract month
        self.reference_start_date = get_nth_weekday_of_month(year, month, 3, 2)  # 3rd Wednesday
        self.reference_end_date = next_imm_date(self.reference_start_date) - dt.timedelta(days=1)

    def reference_array(self):
        dates = _SIFMA_.biz_date_range(self.reference_start_date, self.reference_end_date)
        if self.reference_start_date not in dates:
            dates = np.insert(dates, 0 , self.reference_start_date)
        if self.reference_end_date not in dates:
            dates = np.insert(dates, -1, self.reference_end_date)
        return convert_dates(dates)

if __name__ == '__main__':
    # Example for SOFR 1M Futures
    sofr1m = SOFR1MFuture('SERM4')
    print("SOFR 1M Futures:")
    print(f"Ticker: {sofr1m.ticker}")
    print(f"Reference Start Date: {sofr1m.reference_start_date.date()}")
    print(f"Reference End Date: {sofr1m.reference_end_date.date()}")
    print(f"Reference Dates: {sofr1m.reference_array()}")


    # Example for SOFR 3M Futures
    sofr3m = SOFR3MFuture('SFRU4')
    print("\nSOFR 3M Futures:")
    print(f"Ticker: {sofr3m.ticker}")
    print(f"Reference Start Date: {sofr3m.reference_start_date.date()}")
    print(f"Reference End Date: {sofr3m.reference_end_date.date()}")
    print(f"Reference Dates: {sofr3m.reference_array()}")
