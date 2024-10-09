import datetime as dt
import numpy as np
from dateutil.relativedelta import relativedelta
import pandas as pd

# Use 1904 date format
__base_date__ = dt.datetime(1904, 1, 1)

def convert_dates(dates: pd.DatetimeIndex | np.ndarray | list) -> np.ndarray:
    """
    Conversion of a list of datetime into 1904 int format
    :param dates:
    :return:
    """
    # If it's a pandas DatetimeIndex, convert to an array of datetime
    if isinstance(dates, pd.DatetimeIndex):
        dates = dates.to_pydatetime()

    # Convert each date to Excel 1904 integer format
    return np.array([(d - __base_date__).days for d in dates])

def parse_dates(arr: np.ndarray) -> np.ndarray:
    """
    Converts an iterable of ints back into np.ndarray of datetime
    :param arr:
    :return:
    """
    return __base_date__ + np.array([dt.timedelta(days=int(x)) for x in arr])


# Auxiliary functions
def get_nth_weekday_of_month(year: int, month: int, n: int, weekday: int) -> dt.datetime:
    """
    This function gets the nth weekday of a given year, month. Useful for IMM dates.
    :param year:
    :param month:
    :param n:
    :param weekday:
    :return:
    """
    # weekday: Monday=0, Sunday=6
    first_day = dt.datetime(year, month, 1)
    days_until_weekday = (weekday - first_day.weekday() + 7) % 7
    nth_weekday = first_day + pd.Timedelta(days=days_until_weekday) + pd.Timedelta(weeks=n - 1)
    return nth_weekday

def next_imm_date(d: dt.datetime | dt.date):
    """
    This function gives the IMM date 3 months from a given d
    :param d:
    :return:
    """
    # Calculate the third Wednesday three months before the contract month
    next_quarter = d + relativedelta(months=3)
    return get_nth_weekday_of_month(next_quarter.year, next_quarter.month, 3, 2)