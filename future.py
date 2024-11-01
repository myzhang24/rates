import datetime as dt
from dateutil.relativedelta import relativedelta
from functools import lru_cache
from pandas.tseries.offsets import MonthEnd
from date_util import get_nth_weekday_of_month, next_imm_date

########################################################################################################################
# Code conventions
########################################################################################################################
_CODE_TO_MONTH_ = {
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

_MONTH_TO_CODE_ = {v: k for k, v in _CODE_TO_MONTH_.items()}

_QUARTERLY_CODE_TO_MONTH_ = {
        'H': 3,  # March
        'M': 6,  # June
        'U': 9,  # September
        'Z': 12  # December
    }

########################################################################################################################
# Future utilities
########################################################################################################################
@lru_cache
def parse_future_ticker(ticker: str,
                        ref_date=dt.datetime.now()) -> (str, dt.datetime, dt.datetime):
    ticker = ticker.upper()
    # Ticker format: SER + Month Code + Year Digit(s)
    if ticker.startswith('SER') or ticker.startswith('FF'):
        code = ticker[3:] if ticker.startswith('SER') else ticker[2:]
        month_code = code[0]
        year_code = code[1:]
        if month_code not in _CODE_TO_MONTH_:
            raise ValueError("Invalid month code in ticker.")
        # Handle one or two-digit year codes (e.g., '5' for 2005 or '25' for 2025)
        if len(year_code) == 1:
            current_year = ref_date.year
            current_decade = current_year - current_year % 10
            year_digit = int(year_code)
            year = current_decade + year_digit
            if year < current_year - 5:
                year += 10  # Adjust for decade rollover
        else:
            year = int(year_code) + 2000
        month = _CODE_TO_MONTH_[month_code]

        # 1M futures reference to first to last calendar day
        start_date = dt.datetime(year, month, 1)
        end_date = (start_date + MonthEnd(0)).to_pydatetime()
        fut_type = "SOFR1M" if ticker.startswith('SER') else "FF"
        return fut_type, start_date, end_date

    if not ticker.startswith('SFR') or ticker.startswith("ED"):
        raise ValueError("Invalid ticker format for SOFR 3M Futures.")
    code = ticker[3:]
    month_code = code[0]
    year_code = code[1:]
    if month_code not in _QUARTERLY_CODE_TO_MONTH_:
        raise ValueError("Invalid month code in ticker.")
    # Handle one or two-digit year codes
    if len(year_code) == 1:
        year = 2020 + int(year_code)
    elif len(year_code) == 2:
        year = 2000 + int(year_code)
    else:
        raise ValueError("Invalid year code in ticker.")
    month = _QUARTERLY_CODE_TO_MONTH_[month_code]

    # SOFR 3M Futures expire on the third Wednesday of the contract month
    start_date = get_nth_weekday_of_month(year, month, 3, 2)  # 3rd Wednesday
    end_date = next_imm_date(start_date + dt.timedelta(days=1), False)
    fut_type = "SOFR3M" if ticker.startswith('SFR') else "ED"
    return fut_type, start_date, end_date


@lru_cache
def live_futures(date: dt.datetime, future: str = "sofr3m"):
    """
    Generates live future tickers (BBG) as of a given date
    :param date:
    :param future:
    :return:
    """
    live_tickers = []
    if future.lower() == "sofr3m":
        start = date if date == next_imm_date(date, False) else next_imm_date(date - relativedelta(months=3), False)
        for i in range(16):
            month_code = _MONTH_TO_CODE_[start.month]
            live_tickers.append(f"SFR{month_code}{str(start.year)[-1]}")
            start = next_imm_date(start + relativedelta(months=1), False)

    if future.lower() == "ff" or future.lower() == "sofr1m":
        prefix = "FF" if future.lower() == "ff" else "SER"
        for i in range(12):
            year = date.year + ((date.month + i - 1) // 12)
            month = (date.month + i - 1) % 12 + 1
            month_code = _MONTH_TO_CODE_[month]
            live_tickers.append(f"{prefix}{month_code}{str(year)[-1]}")

    return live_tickers

########################################################################################################################
# IRFuture class
########################################################################################################################
class IRFuture:
    def __init__(self, ticker):
        self.ticker = ticker.upper()
        self.reference_start_date = None
        self.reference_end_date = None
        self.future_type = None
        self.future_type, self.reference_start_date, self.reference_end_date = parse_future_ticker(self.ticker)

    def get_reference_start_end_dates(self):
        return [self.reference_start_date, self.reference_end_date]

########################################################################################################################
# debug
########################################################################################################################
def debug_parsing():
    # Example for SOFR 1M Futures
    sofr1m = IRFuture('SERM4')
    print("SOFR 1M Futures:")
    print(f"Ticker: {sofr1m.ticker}")
    print(f"Reference Start Date: {sofr1m.reference_start_date.date()}")
    print(f"Reference End Date: {sofr1m.reference_end_date.date()}")

    # Example for SOFR 3M Futures
    sofr3m = IRFuture('SFRU4')
    print("\nSOFR 3M Futures:")
    print(f"Ticker: {sofr3m.ticker}")
    print(f"Reference Start Date: {sofr3m.reference_start_date.date()}")
    print(f"Reference End Date: {sofr3m.reference_end_date.date()}")

def debug_generation():
    date = dt.datetime(2024, 9, 13)
    ff = live_futures(date, "FF")
    sofr1m = live_futures(date, "SOFR1M")
    sofr3m = live_futures(date, "SOFR3M")
    print(f"\nAs of {date.date()} the live futures are")
    print(ff)
    print(sofr1m)
    print(sofr3m)

if __name__ == '__main__':
    debug_parsing()
    debug_generation()
    exit(0)