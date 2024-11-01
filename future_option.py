import datetime as dt
import numpy as np
import pandas as pd
from copy import deepcopy
import re
from dateutil.relativedelta import relativedelta
from scipy.interpolate import CubicSpline
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

from future import _MONTH_TO_CODE_, _CODE_TO_MONTH_, IRFuture
from date_util import get_nth_weekday_of_month, next_imm_date, day_count
from curve import USDCurve
from math_util import _implied_normal_vol, _normal_greek, _normal_price

########################################################################################################################
# Options utilities
########################################################################################################################
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

def parse_sofr_option_ticker(bbg_ticker: str):
    """
    Returns chain, expiry code, underlying ticker, expiry, call_put_straddle, strike
    For chain code only return chain expiry and underlying ticker
    :param bbg_ticker:
    :return:
    """
    bbg_ticker = bbg_ticker.upper()
    # Try to match the prefix
    prefix = None
    for p in _MIDCURVINESS_.keys():
        if bbg_ticker.startswith(p):
            prefix = p
            break
    if not prefix:
        raise ValueError("Invalid ticker prefix for SOFR options")

    # Extract the rest of the ticker
    rest = re.findall(rf"{prefix}([FGHJKMNQUVXZ]\d+)", bbg_ticker)[0]

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
    chain = prefix + expiry_code

    # Add mid-curviness in months
    months_ahead = _MIDCURVINESS_[prefix]
    underlying_start = next_imm_date(expiry + relativedelta(months=months_ahead), False)
    underlying_code = expiry_to_code(underlying_start)
    underlying_ticker = f"SFR{underlying_code}"

    # Call put and strike
    try:
        call_put, strike = re.findall(r"([CPS]) ([0-9]*[.]?[0-9]+)", bbg_ticker)[0]
        strike = float(strike)
        if call_put == "P":
            cps = -1
        elif call_put == "C":
            cps = 1
        elif call_put == "S":
            cps = 2
        else:
            cps = 0
        return chain, expiry_code, underlying_ticker, expiry, cps, strike
    except:
        return chain, expiry_code, underlying_ticker, expiry

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
    live_options = sorted(live_options, key=lambda x: parse_sofr_option_ticker(x)[-1])
    return live_options

def get_live_expiries(ref_date: dt.datetime, future_ticker: str) -> list:
    all_options = get_live_sofr_options(ref_date)
    return [opt for opt in all_options if parse_sofr_option_ticker(opt)[2] == future_ticker.upper()]

def parse_option_data(market_data: pd.Series, curve: USDCurve) -> dict:
    df = pd.DataFrame(market_data, columns=["premium"]).rename_axis("ticker")
    for bbg_ticker in market_data.index:
        try:
            chain, expiry_ticker, underlying_ticker, expiry_datetime, cp, strike = parse_sofr_option_ticker(bbg_ticker)
            df.loc[bbg_ticker, "chain"] = chain
            df.loc[bbg_ticker, "expiry"] = expiry_ticker
            df.loc[bbg_ticker, "expiry_dt"] = expiry_datetime
            df.loc[bbg_ticker, "underlying_ticker"] = underlying_ticker
            df.loc[bbg_ticker, "cp"] = cp
            df.loc[bbg_ticker, "strike"] = strike
        except:
            continue
    df = df.dropna()
    gb = df.reset_index().groupby("chain")
    df_dict = {key: group.set_index("ticker") for key, group in gb}
    for chain, df in df_dict.items():
        # Get expiry and ticker
        exp_dt = df["expiry_dt"].iloc[0]
        fut_ticker = df["underlying_ticker"].iloc[0]

        # Here future price is either read from the curve market instruments, and if not present then priced from curve
        if fut_ticker in curve.market_data["SOFR3M"].index:
            fut_price = curve.market_data["SOFR3M"][fut_ticker]
        else:
            fut_price = curve.price_3m_futures([fut_ticker]).squeeze()
        df["underlying_price"] = fut_price

        # Add OTM column
        otm_rows = df["cp"] == 2
        otm_rows |= (df["cp"] == 1) & (df["strike"] >= df["underlying_price"])
        otm_rows |= (df["cp"] == -1) & (df["strike"] <= df["underlying_price"])
        df["otm"] = otm_rows

        # Add pricing columns
        t2e = day_count(curve.reference_date, exp_dt + dt.timedelta(days=1), "BIZ/252")
        df["t2e"] = t2e # Add 1 because expires at end of day
        disc = curve.future_discount_factor(exp_dt)    # discount until expiry day but not overnight, computed from curve
        df["disc"] = disc
    return df_dict

########################################################################################################################
# Option and Vol class
########################################################################################################################
class IROption:
    def __init__(self, bbg_ticker: str):
        self.bbg_ticker = bbg_ticker.upper()
        self.chain, self.expiry_ticker, underlying_ticker, self.expiry_datetime, self.cp, self.strike = parse_sofr_option_ticker(bbg_ticker)
        self.underlying = IRFuture(underlying_ticker)

class SOFRFutureOptionVolGrid:
    def __init__(self, curve: USDCurve):
        self.curve = curve
        self.reference_date = curve.reference_date
        self.option_data = {}
        self.vol_grid = {}

    def load_option_data(self, market_data: pd.Series):
        self.option_data = parse_option_data(market_data, self.curve)
        for key, df in self.option_data.items():
            # Solve for vol
            df["vol"] = _implied_normal_vol(df["disc"].values.squeeze(),
                                            df["t2e"].values.squeeze(),
                                            df["underlying_price"].values.squeeze(),
                                            df["strike"].values.squeeze(),
                                            df["premium"].values.squeeze(),
                                            df["cp"].values.squeeze())

            # Compute greeks
            d, g, v, t = _normal_greek(df["disc"].values.squeeze(),
                                       df["t2e"].values.squeeze(),
                                       df["underlying_price"].values.squeeze(),
                                       df["strike"].values.squeeze(),
                                       df["vol"].values.squeeze(),
                                       df["cp"].values.squeeze())
            df["delta"] = d
            df["gamma"] = g
            df["vega"] = v
            df["theta"] = t * 1 / 252
        return self

    def calibrate_vol_grid(self, method="cubic"):
        if method == "cubic":
            for key, vols in self.option_data.items():
                otm_vol = vols.loc[vols["otm"] == 1, ["strike", "vol"]].sort_values("strike")
                cb = CubicSpline(otm_vol["strike"], otm_vol["vol"], bc_type="natural")
                self.vol_grid[key] = cb
        if method == "SABR":
            pass
        return self

    def vol_grid_price(self, tickers: list | pd.Series):
        pass

    def atm(self, chain: str) -> (float, float):
        """
        Given a chain, return the ATM vol (as an interpolation)
        :param chain:
        :return:
        """
        _, _, underlying, _ = parse_sofr_option_ticker(chain)
        if underlying in self.curve.market_data.get("SOFR3M", pd.Series()).index:
            fut_price = self.curve.market_data["SOFR3M"][underlying]
        else:
            fut_price = self.curve.price_3m_futures([underlying]).squeeze()
        atm_vol = self.vol_grid[chain](fut_price)
        return fut_price, atm_vol

    def plot_smile(self, chain: str):
        """
        Plots market vol smile vs vol_grid smile
        :param chain:
        :return:
        """
        # Get ATM
        fut_price, atm_vol = self.atm(chain)

        # Get Market
        mkt = self.option_data[chain]
        mkt = mkt.loc[mkt["otm"] == 1, ["strike", "vol"]].sort_values("strike").set_index("strike")

        # Get Vol Grid
        strikes = np.linspace(mkt.index.min(), mkt.index.max(), 50)
        vols = self.vol_grid[chain](strikes)
        grid = pd.DataFrame(vols, index=strikes)

        # Plotting the line plot for grid with specified style (thin dotted opaque blue)
        plt.figure()
        plt.plot(grid.index, grid.values, linestyle=':', linewidth=1, color=(0, 0, 1, 0.5), label='Vol Grid')

        # Overlaying scatter plot for mkt (black dots, size 2)
        plt.scatter(mkt.index, mkt.values, color='black', s=5, label='Market OTM Vol')

        # Add ATM dot
        plt.scatter(np.array([fut_price]), np.array([atm_vol]), color="red", s=10, label="ATM Vol")

        # Adding a solid red vertical line at x=fut_price
        plt.axvline(x=fut_price, color='red', linestyle='-', linewidth=1, label='Future Price', alpha=0.5)

        # Adding a label near the ATM vol point
        plt.text(fut_price, atm_vol-0.05, f'ATM: ({fut_price:.3f}, {atm_vol:.1%})', color='red', fontsize=9, ha='left',
                 va='bottom')

        # Formatting the y-axis for major grid and labels (100 * {y}.0f format)
        plt.yticks(
            ticks=[i / 20 for i in range(int(grid.values.min() * 20), int(grid.values.max() * 20) + 1)],
            labels=[f'{100 * y:.0f}' for y in
                    [i / 20 for i in range(int(grid.values.min() * 20), int(grid.values.max() * 20) + 1)]]
        )
        plt.gca().yaxis.grid(True, which='major', linestyle='--', linewidth=0.5)

        # Formatting the x-axis for major and minor grids
        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1.0))  # Major grid at 1.0 intervals
        plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(0.125))  # Minor grid at 0.125 intervals

        # Setting formatters for the x-axis ticks
        plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))  # Major ticks formatted as integer
        plt.gca().xaxis.set_minor_formatter(ticker.FuncFormatter(
            lambda x, _: f'.{str(x % 1.0)[2:].rstrip("0")}' if x % 1.0 != 0 else ''
            # Minor ticks formatted with leading dot and no trailing zeros
        ))

        # Rotate minor tick labels and set font sizes
        plt.tick_params(axis='x', which='major', labelsize=10)  # Larger font for major ticks
        plt.tick_params(axis='x', which='minor', labelsize=8, rotation=90)  # Smaller font and rotated for minor ticks

        plt.gca().xaxis.grid(True, which='major', linestyle='--', linewidth=0.5)
        plt.gca().xaxis.grid(True, which='minor', linestyle=':', linewidth=0.25)

        # Labels and legend
        plt.xlabel('Strike')
        plt.ylabel('Normal Biz Vol (%)')
        plt.title(f"Vol smile for {chain} as of {self.reference_date.date()}")
        plt.legend()

        # Display the plot
        plt.tight_layout()
        plt.show()

def shock_surface_by_curve(vol_grid: SOFRFutureOptionVolGrid,
                           new_curve: USDCurve,
                           backbone: float = 0,
                           new_surface: bool = True):
    """
    Shock a vol_grid with a rate scenario in a sticky delta fashion -> same delta options have same vol
    :param new_surface:
    :param backbone:
    :param vol_grid:
    :param new_curve:
    :return:
    """
    if new_surface:
        vol_grid = deepcopy(vol_grid)
    for chain, df in vol_grid.option_data.items():
        # Get new underlying price
        _, _, underlying_ticker, _ = parse_sofr_option_ticker(chain)
        if underlying_ticker in new_curve.market_data["SOFR3M"].index:
            underlying_price_new = new_curve.market_data["SOFR3M"][underlying_ticker]
        else:
            underlying_price_new = new_curve.price_3m_futures([underlying_ticker]).squeeze()

        # Get old underlying price and atm vol
        underlying_price_old, atm_vol_old = vol_grid.atm(chain)
        price_shift = underlying_price_new - underlying_price_old
        vol_shift = 1e2 * price_shift * backbone

        # Shift the vol
        df["vol"] = vol_grid.vol_grid[chain](df["strike"] - price_shift) + vol_shift
        df["underlying_price"] = df["underlying_price"] + price_shift
        df["otm"] = np.where(df["cp"] == 2, True,
                             np.where(df["cp"] == 1, df["strike"] >= df["underlying_price"],
                                      df["strike"] <= df["underlying_price"]))
        df["premium"] = _normal_price(df["disc"].values.squeeze(),
                                      df["t2e"].values.squeeze(),
                                      df["underlying_price"].values.squeeze(),
                                      df["strike"].values.squeeze(),
                                      df["vol"].values.squeeze(),
                                      df["cp"].values.squeeze())
        # Compute greeks
        d, g, v, t = _normal_greek(df["disc"].values.squeeze(),
                                   df["t2e"].values.squeeze(),
                                   df["underlying_price"].values.squeeze(),
                                   df["strike"].values.squeeze(),
                                   df["vol"].values.squeeze(),
                                   df["cp"].values.squeeze())
        df["delta"] = d
        df["gamma"] = g
        df["vega"] = v
        df["theta"] = t * 1 / 252

    vol_grid.curve = new_curve
    vol_grid.calibrate_vol_grid()
    return vol_grid


if __name__ == '__main__':
    pass