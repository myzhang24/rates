import datetime as dt
from dateutil.relativedelta import relativedelta
from futures import _MONTH_CODES_
from date_utils import get_nth_weekday_of_month, next_imm_date

_QUARTERLY_CODE_ = {3: "H", 6: "M", 9: "U", 12: "Z"}

def parse_sofr_option_ticker(ticker: str):
    ticker = ticker.upper()
    # Define valid prefixes
    prefixes = ['SFR', '0Q', '2Q', '3Q', '4Q', '5Q']

    # Try to match the prefix
    prefix = None
    for p in prefixes:
        if ticker.startswith(p):
            prefix = p
            break

    if not prefix:
        raise ValueError("Invalid ticker prefix for SOFR options")

    # Extract the rest of the ticker
    rest = ticker[len(prefix):]

    if len(rest) not in [2, 3]:
        raise ValueError("Invalid ticker format")

    month_code = rest[0]
    year_code = rest[1:]

    if month_code not in _MONTH_CODES_:
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

    month = _MONTH_CODES_[month_code]
    # Options expire the friday before the 3rd wednesday of the contract month
    imm_date = get_nth_weekday_of_month(year, month, 3, 2)
    expiry = imm_date - dt.timedelta(days=5)
    expiry = expiry.replace(hour=15) # 4pm CT 3pm EST

    years_ahead = 0 if prefix == "SFR" else int(prefix[0]) if prefix[0] != '0' else 1
    underlying_start = next_imm_date(expiry + relativedelta(years=years_ahead), 0)
    underlying_month_code = _QUARTERLY_CODE_[underlying_start.month]
    underlying_ticker = f"SFR{underlying_month_code}{underlying_start.year % 10}"

    return expiry, underlying_ticker

# Example usage:
exp, tick = parse_sofr_option_ticker('SFRZ3')
print(f"Expiry: {exp}, Underlying: {tick}")

exp, tick = parse_sofr_option_ticker('0QZ3')
print(f"Expiry: {exp}, Underlying: {tick}")
