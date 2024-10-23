import pandas as pd
import numpy as np
import datetime as dt

__fixing_cache__ = {}

class FixingManager:
    def __init__(self, rate_name="SOFR"):
        self.rate_name = rate_name.upper()
        self.original_file = f"fixing/{self.rate_name}.csv"
        self.clean_file = f"fixing/{self.rate_name}.pkl"

    def clean_fixings(self):
        df = pd.read_csv(self.original_file, index_col=0)
        df.index = pd.to_datetime(df.index)
        df = df.replace(".", np.nan).dropna().astype(float)
        df.to_pickle(self.clean_file)
        return self

    def load_fixings(self,):
        global __fixing_cache__
        __fixing_cache__[self.rate_name] = pd.read_pickle(self.clean_file)
        return self

    def get_fixings(self, st: dt.datetime, et: dt.datetime):
        if self.rate_name not in __fixing_cache__:
            self.load_fixings()
        return __fixing_cache__[self.rate_name].loc[st: et]

    def get_fixings_asof(self, st: dt.datetime, et: dt.datetime):
        if self.rate_name not in __fixing_cache__:
            self.load_fixings()
        dates = pd.date_range(st, et, freq='D')
        res = pd.DataFrame(index=dates)
        return pd.merge_asof(res, __fixing_cache__[self.rate_name], left_index=True, right_index=True).squeeze()

_SOFR_ = FixingManager("SOFR")
_FF_ = FixingManager("FF")

if __name__ == '__main__':

    # Clean fixing
    _SOFR_.clean_fixings()
    _FF_.clean_fixings()

    # Load fixing
    dg = _SOFR_.get_fixings(dt.datetime(2022, 1, 1), dt.datetime(2024, 10, 8))
    exit(0)
