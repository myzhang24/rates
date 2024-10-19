import datetime as dt
import re
from dateutil.relativedelta import relativedelta

from future import _MONTH_TO_CODE_, _CODE_TO_MONTH_, IRFuture
from date_util import get_nth_weekday_of_month, next_imm_date
from curve import SOFRCurve

_MIDCURVINESS_ = {
    'SFR': 0,  # Standard
    "SRA": 3,  # 3m mid curves
    "SRR": 6,  # 6m mid curves
    "SRW": 9,  # 9m mid curves
    '0Q': 12,  # 1Y mid curves
    '2Q': 24,  # 2Y mid curves
    '3Q': 36,  # 3Y mid curves
    '4Q': 48,  # 4Y mid curves
    '5Q': 60,  # 5Y mid curves
}

def expiry_to_code(d: dt.datetime | dt.date) -> str:
    month_code = _MONTH_TO_CODE_[d.month]
    year_code = str(d.year % 100)
    return f"{month_code}{year_code}"

def parse_sofr_option_ticker(ticker: str):
    ticker = ticker.upper()
    # Try to match the prefix
    prefix = None
    for p in _MIDCURVINESS_.keys():
        if ticker.startswith(p):
            prefix = p
            break
    if not prefix:
        raise ValueError("Invalid ticker prefix for SOFR options")

    # Extract the rest of the ticker
    rest = re.findall(rf"{prefix}([FGHJKMNQUVXZ]\d+)", ticker)[0]

    if len(rest) not in [2, 3]:
        raise ValueError("Invalid ticker format")

    month_code = rest[0]
    year_code = rest[1:]

    if month_code not in _CODE_TO_MONTH_:
        raise ValueError("Invalid month code")

    # Convert year code to full year (handles decade rollover)
    if len(year_code) == 1:
        current_year = dt.datetime.now().year
        current_decade = current_year - current_year % 10
        year_digit = int(year_code)
        year = current_decade + year_digit
        if year < current_year - 5:
            year += 10  # Adjust for decade rollover
    else:
        year = int(year_code) + 2000

    month = _CODE_TO_MONTH_[month_code]
    # Options expire the friday before the 3rd wednesday of the contract month
    imm_date = get_nth_weekday_of_month(year, month, 3, 2)
    expiry = imm_date - dt.timedelta(days=5)
    expiry = expiry.replace(hour=15) # 4pm CT 3pm EST

    # Add mid-curviness in months
    months_ahead = _MIDCURVINESS_[prefix]
    underlying_start = next_imm_date(expiry + relativedelta(months=months_ahead), False)
    underlying_code = expiry_to_code(underlying_start)
    underlying_ticker = f"SFR{underlying_code}"

    # Call put and strike
    try:
        cp, strike = re.findall(r"([CP]) ([0-9]*[.]?[0-9]+)", ticker)[0]
        strike = float(strike)
    except:
        return expiry, underlying_ticker
    return expiry, underlying_ticker, cp, strike

def get_live_sofr_options(reference_date: dt.datetime):
    """
    Get live listed SOFR options according to CME listing schedule
    https://www.cmegroup.com/education/articles-and-reports/trading-sofr-options.html
    :param reference_date:
    :return:
    """
    # Generate 16 consecutive quarterly months
    quarterly_expiry = []
    date = reference_date
    while len(quarterly_expiry) < 16:
        date = next_imm_date(date, False)
        quarterly_expiry.append(expiry_to_code(date))
        date += dt.timedelta(days=1)

    # Generate nearest 4 non-quarterly months
    serial_expiry = []
    date = reference_date
    while len(serial_expiry) < 4:
        date = next_imm_date(date, True)
        if date.month in [3, 6, 9, 12]:
            date += dt.timedelta(days=1)
            continue
        serial_expiry.append(expiry_to_code(date))
        date += dt.timedelta(days=1)

    # All 16 quarterly expiry and all 4 serial expiry
    regular_options = [f"SFR{x}" for x in quarterly_expiry] + [f"SFR{x}" for x in serial_expiry]

    # 2. First 5 quarterly expiry and all 4 serial expiry for 1y, 2y, 3y, 4y, 5y Mid-Curve Options
    midcurve_options = []
    prefixes = ['0Q', '2Q', '3Q', '4Q', '5Q']
    for prefix in prefixes:
        midcurve_options += [f"{prefix}{x}" for x in quarterly_expiry[:5]]
        midcurve_options += [f"{prefix}{x}" for x in serial_expiry]

    # 3. First quarterly expiry and 2 serial expiry for 3M, 6M, 9M Mid-Curve Options
    prefixes = ['SRA', 'SRR', 'SRW']
    for prefix in prefixes:
        midcurve_options += [f"{prefix}{quarterly_expiry[0]}",
                             f"{prefix}{serial_expiry[0]}",
                             f"{prefix}{serial_expiry[1]}"]

    live_options = regular_options + midcurve_options
    live_options = sorted(live_options, key=lambda x: parse_sofr_option_ticker(x)[1])
    return live_options

def get_live_expiries(ref_date: dt.datetime, future_ticker: str) -> list:
    all_options = get_live_sofr_options(ref_date)
    return [opt for opt in all_options if parse_sofr_option_ticker(opt)[1] == future_ticker.upper()]


class IROption:
    def __init__(self, ticker: str):
        ticker = ticker.upper()
        self.ticker = ticker
        self.expiry, underlying_ticker, self.cp, self.strike = parse_sofr_option_ticker(ticker)
        self.underlying = IRFuture(underlying_ticker)


class SOFRFutureOptionVolGrid:
    def __init__(self, curve: SOFRCurve):
        self.curve = curve
        self.reference_date = curve.reference_date


def debug_parsing():
    # Example usage:
    exp, tick = parse_sofr_option_ticker('SFRH25')
    print(f"Expiry: {exp}, Underlying: {tick}")

    exp, tick = parse_sofr_option_ticker('0QZ23')
    print(f"Expiry: {exp}, Underlying: {tick}")

    # Generate tickers as of today
    today = dt.datetime.now()
    live = get_live_sofr_options(today)
    print(f"Today's live SOFR options: {live}")

    exp = get_live_expiries(today, "SFRZ24")
    print(f"Today's expiries for SFRZ24 futures: {exp}")

def debug_option_pricing():
    import pandas as pd
    from curve import SOFRCurve
    sofr3m = pd.Series({
        "SFRU4": 95.2125,
        "SFRZ4": 95.615,
        "SFRH5": 96.025,
        "SFRM5": 96.33,
        "SFRU5": 96.505,
        "SFRZ5": 96.595,
        "SFRH6": 96.64,
        "SFRM6": 96.655,
        "SFRU6": 96.645,
    })
    sofr1m = pd.Series({
        "SERV4": 95.1525,
        "SERX4": 95.32,
        "SERZ4": 95.425,
        "SERF5": 95.555,
        "SERG5": 95.735,
        "SERH5": 95.82,
        "SERJ5": 95.965,
        "SERK5": 96.10,
        "SERM5": 96.195,
        "SERN5": 96.30,
        "SERQ5": 96.39,
        "SERU5": 96.42,
        "SERV5": 96.48,
    })
    ref_date = dt.datetime(2024, 10, 18)
    sofr = SOFRCurve(ref_date).calibrate_futures_curve(sofr1m, sofr3m)
    sofr.plot_futures_daily_forwards(16, 6)
    exit(0)

if __name__ == '__main__':
    # debug_parsing()

    debug_option_pricing()
    exit(0)