import datetime as dt
import pandas as pd
import re
from dateutil.relativedelta import relativedelta

from future import _MONTH_TO_CODE_, _CODE_TO_MONTH_, IRFuture
from date_util import get_nth_weekday_of_month, next_imm_date, day_count
from curve import SOFRCurve, price_3m_futures
from math_util import _implied_normal_vol, _normal_greek

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
    year_code = str(d.year % 10)
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
    expiry_code = expiry_to_code(expiry)

    # Add mid-curviness in months
    months_ahead = _MIDCURVINESS_[prefix]
    underlying_start = next_imm_date(expiry + relativedelta(months=months_ahead), False)
    underlying_code = expiry_to_code(underlying_start)
    underlying_ticker = f"SFR{underlying_code}"

    # Call put and strike
    try:
        call_put, strike = re.findall(r"([CPS]) ([0-9]*[.]?[0-9]+)", ticker)[0]
        strike = float(strike)
        if call_put == "P":
            cps = -1
        if call_put == "C":
            cps = 1
        if call_put == "S":
            cps = 2
    except:
        return expiry, underlying_ticker
    return expiry_code, underlying_ticker, expiry, cps, strike

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
    live_options = sorted(live_options, key=lambda x: parse_sofr_option_ticker(x)[0])
    return live_options

def get_live_expiries(ref_date: dt.datetime, future_ticker: str) -> list:
    all_options = get_live_sofr_options(ref_date)
    return [opt for opt in all_options if parse_sofr_option_ticker(opt)[1] == future_ticker.upper()]


class IROption:
    def __init__(self, ticker: str):
        ticker = ticker.upper()
        self.ticker = ticker
        self.expiry_ticker, underlying_ticker, self.expiry_datetime, self.cp, self.strike = parse_sofr_option_ticker(ticker)
        self.underlying = IRFuture(underlying_ticker)


class SOFRFutureOptionVolGrid:
    def __init__(self, curve: SOFRCurve):
        self.curve = curve
        self.reference_date = curve.reference_date
        self.fut_data = pd.Series()
        self.option_data = {}

    def load_option_data(self, market_data: pd.Series, sofr3m: pd.Series=pd.Series(), sofr1m: pd.Series=pd.Series()):
        if not sofr3m.empty:
            self.fut_data = sofr3m
        # Need to recalibrate curve to future data
        if not sofr1m.empty:
            self.curve = self.curve.calibrate_futures_curve(sofr1m, sofr3m)
        else:
            self.curve = self.curve.calibrate_futures_curve_3m(sofr3m)
        self.option_data = self.parse_option_data(market_data)
        return self

    def parse_option_data(self, market_data: pd.Series) -> dict:
        df = pd.DataFrame(market_data, columns=["premium"]).rename_axis("ticker")
        for ticker in market_data.index:
            try:
                expiry_ticker, underlying_ticker, expiry_datetime, cp, strike = parse_sofr_option_ticker(ticker)
                df.loc[ticker, "expiry"] = expiry_ticker
                df.loc[ticker, "expiry_dt"] = expiry_datetime
                df.loc[ticker, "underlying_ticker"] = underlying_ticker
                df.loc[ticker, "cp"] = cp
                df.loc[ticker, "strike"] = strike
            except:
                continue
        df = df.dropna()
        gb = df.reset_index().groupby(["expiry", "underlying_ticker"])
        df_dict = {key: group.set_index("ticker").drop(columns=["expiry", "underlying_ticker"]) for key, group in gb}
        for key, df in df_dict.items():
            exp_dt = df["expiry_dt"].iloc[0]
            fut_ticker = key[1]
            if fut_ticker in self.fut_data.index:
                fut_price = self.fut_data[fut_ticker]
            else:
                fut_price = price_3m_futures(self.curve, [fut_ticker]).squeeze()
            df["underlying_price"] = fut_price
            t2e = day_count(self.reference_date, exp_dt + dt.timedelta(days=1), "BIZ/252")
            df["t2e"] = t2e # Add 1 because expires at end of day
            disc = self.curve.future_discount_factor(exp_dt)    # discount until expiry day but not overnight
            df["disc"] = disc
            df["vol"] = _implied_normal_vol(disc, t2e, fut_price,
                                            df["strike"].values.squeeze(),
                                            df["premium"].values.squeeze(),
                                            df["cp"].values.squeeze())
            d, g, v, t = _normal_greek(disc, t2e, fut_price,
                                       df["strike"].values.squeeze(),
                                       df["vol"].values.squeeze(),
                                       df["cp"].values.squeeze())
            df["delta"] = d
            df["gamma"] = g
            df["vega"] = v
            df["theta"] = t * 1/252
        return df_dict

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
        "SFRU4": 95.2175,
        "SFRZ4": 95.635,
        "SFRH5": 96.05,
        "SFRM5": 96.355,
        "SFRU5": 96.53,
        "SFRZ5": 96.625,
        "SFRH6": 96.67,
        "SFRM6": 96.685,
        "SFRU6": 96.675,
    })
    sofr1m = pd.Series({
        "SERV4": 95.1525,
        "SERX4": 95.325,
        "SERZ4": 95.435,
    })
    ref_date = dt.datetime(2024, 10, 18)
    # sofr = SOFRCurve(ref_date).calibrate_futures_curve_3m(sofr3m)
    sofr = SOFRCurve(ref_date).calibrate_futures_curve(sofr1m, sofr3m)
    market_data = pd.Series({
        "SFRZ4C 95.25":	    0.3875,
        "SFRZ4C 95.375":	0.27,
        "SFRZ4C 95.50":	    0.1725,
        "SFRZ4C 95.625":	0.095,
        "SFRZ4C 95.75":	    0.0475,
        "SFRZ4C 95.875":	0.03,
        "SFRZ4C 96.00":	    0.02,
        "SFRZ4C 96.125":	0.015,
        "SFRZ4C 96.25":	    0.01,
        "SFRZ4P 95.875":	0.2675,
        "SFRZ4P 95.750":	0.1625,
        "SFRZ4P 95.625":	0.085,
        "SFRZ4P 95.500":	0.0375,
        "SFRZ4P 95.375":	0.0125,
        "SFRZ4P 95.250":	0.005,
    })
    vol_grid = SOFRFutureOptionVolGrid(sofr)
    vol_grid.load_option_data(market_data, sofr3m)
    exit(0)

if __name__ == '__main__':
    # debug_parsing()

    debug_option_pricing()
    exit(0)