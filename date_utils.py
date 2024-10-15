import datetime as dt
import numpy as np
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    Holiday,
    nearest_workday,
    USMemorialDay,
    USLaborDay,
    USMartinLutherKingJr,
    USPresidentsDay,
    USThanksgivingDay,
    GoodFriday,
    USColumbusDay
)
from dateutil.relativedelta import relativedelta
import time
import logging
# Ensure logging is configured
logging.basicConfig(level=logging.INFO)
from functools import wraps


class SIFMACalendar(AbstractHolidayCalendar):
    rules = [
        Holiday('NewYearsDay', month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday('Juneteenth', month=6, day=19, observance=nearest_workday, start_date='2022-01-01'),
        Holiday('IndependenceDay', month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USColumbusDay,
        Holiday('VeteransDay', month=11, day=11, observance=nearest_workday),
        USThanksgivingDay,
        Holiday('ChristmasDay', month=12, day=25, observance=nearest_workday),
    ]

    def __init__(self):
        super().__init__()
        self.holiday_set = self.holidays(start='1990-01-01', end='2060-12-31')

    def is_biz_day(self, d: dt.datetime | dt.date) -> bool:
        return d.weekday() < 5 and d not in self.holiday_set

    def prev_biz_day(self, d: dt.datetime | dt.date, shift=1) -> dt.datetime | dt.date:
        d -= dt.timedelta(days=shift)
        while not self.is_biz_day(d):
            d -= dt.timedelta(days=1)
        return d

    def next_biz_day(self, d: dt.datetime | dt.date, shift=1) -> dt.datetime | dt.date:
        d += dt.timedelta(days=shift)
        while not self.is_biz_day(d):
            d += dt.timedelta(days=1)
        return d

    def biz_date_range(self, st: dt.datetime | dt.date, et: dt.datetime | dt.date) -> pd.Series:
        dates = pd.date_range(start=st, end=et, freq='D')
        biz_days = pd.to_datetime([dt for dt in dates if self.is_biz_day(dt)])
        return biz_days

_SIFMA_ = SIFMACalendar()


def adjust_date(date, convention):
    if convention == 'Following':
        adjusted_date = _SIFMA_.next_biz_day(date, 0)
    elif convention == 'Modified Following':
        adjusted_date = modified_following(date)
    elif convention == 'Preceding':
        adjusted_date = _SIFMA_.prev_biz_day(date, 0)
    elif convention == 'Modified Preceding':
        adjusted_date = modified_preceding(date)
    else:
        adjusted_date = date
    return adjusted_date


def modified_following(date):
    candidate = _SIFMA_.next_biz_day(date, 0)
    if candidate.month != date.month:
        return _SIFMA_.prev_biz_day(date, 0)
    return candidate


def modified_preceding(date):
    candidate = _SIFMA_.prev_biz_day(date, 0)
    if candidate.month != date.month:
        return _SIFMA_.next_biz_day(date, 0)
    return candidate

# Use 1904 date format
__base_date__ = dt.datetime(1904, 1, 1)

def convert_date(dates: dt.datetime | dt.date | pd.DatetimeIndex | np.ndarray | list):
    """
    Conversion of a datetime or a list of datetime into 1904 int format
    :param dates:
    :return:
    """
    # If it is a single date, return int
    if isinstance(dates, dt.datetime) or isinstance(dates, dt.date):
        return (dates - __base_date__).days

    # If it's a pandas DatetimeIndex, convert to an array of datetime
    if isinstance(dates, pd.DatetimeIndex):
        dates = dates.to_pydatetime()

    # Convert each date to Excel 1904 integer format
    return np.array([(d - __base_date__).days for d in dates])

def parse_date(arr: int| float | np.ndarray) -> dt.datetime | np.ndarray:
    """
    Converts an iterable of ints back into np.ndarray of datetime
    :param arr:
    :return:
    """
    if isinstance(arr, np.ndarray | list | pd.DatetimeIndex | pd.Series):
        return __base_date__ + np.array([dt.timedelta(days=int(x)) for x in arr])
    return __base_date__ + dt.timedelta(days=int(arr))



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

__FOMC_Meetings__ = [
        # 2019
        dt.datetime(2019, 1, 30),
        dt.datetime(2019, 3, 20),
        dt.datetime(2019, 5, 1),
        dt.datetime(2019, 6, 19),
        dt.datetime(2019, 7, 31),
        dt.datetime(2019, 9, 18),
        dt.datetime(2019, 10, 30),
        dt.datetime(2019, 12, 11),
        # 2020
        dt.datetime(2020, 1, 29),
        dt.datetime(2020, 3, 18),
        dt.datetime(2020, 4, 29),
        dt.datetime(2020, 6, 10),
        dt.datetime(2020, 7, 29),
        dt.datetime(2020, 9, 16),
        dt.datetime(2020, 11, 5),
        dt.datetime(2020, 12, 16),
        # 2021
        dt.datetime(2021, 1, 27),
        dt.datetime(2021, 3, 17),
        dt.datetime(2021, 4, 28),
        dt.datetime(2021, 6, 16),
        dt.datetime(2021, 7, 28),
        dt.datetime(2021, 9, 22),
        dt.datetime(2021, 11, 3),
        dt.datetime(2021, 12, 15),
        # 2022
        dt.datetime(2022, 1, 26),
        dt.datetime(2022, 3, 16),
        dt.datetime(2022, 5, 4),
        dt.datetime(2022, 6, 15),
        dt.datetime(2022, 7, 27),
        dt.datetime(2022, 9, 21),
        dt.datetime(2022, 11, 2),
        dt.datetime(2022, 12, 14),
        # 2023
        dt.datetime(2023, 2, 1),
        dt.datetime(2023, 3, 22),
        dt.datetime(2023, 5, 3),
        dt.datetime(2023, 6, 14),
        dt.datetime(2023, 7, 26),
        dt.datetime(2023, 9, 20),
        dt.datetime(2023, 11, 1),
        dt.datetime(2023, 12, 13),
        # 2024
        dt.datetime(2024, 1, 31),
        dt.datetime(2024, 3, 20),
        dt.datetime(2024, 5, 1),
        dt.datetime(2024, 6, 12),
        dt.datetime(2024, 7, 31),
        dt.datetime(2024, 9, 18),
        dt.datetime(2024, 11, 7),
        dt.datetime(2024, 12, 18),
        # 2025
        dt.datetime(2025, 1, 29),
        dt.datetime(2025, 3, 19),
        dt.datetime(2025, 5, 7),
        dt.datetime(2025, 6, 18),
        dt.datetime(2025, 7, 30),
        dt.datetime(2025, 9, 17),
        dt.datetime(2025, 10, 29),
        dt.datetime(2025, 12, 10),
        # 2026
        dt.datetime(2026, 1, 28),
        dt.datetime(2026, 3, 18),
        dt.datetime(2026, 4, 29),
        dt.datetime(2026, 6, 17),
        dt.datetime(2026, 7, 29),
        dt.datetime(2026, 9, 16),
        dt.datetime(2026, 10, 28),
        dt.datetime(2026, 12, 9),
]

def generate_fomc_meeting_dates(start_date: dt.datetime, end_date: dt.datetime):
    """
    Generates FOMC meeting dates within the given date range.

    Args:
        start_date (datetime.date): The start date of the range.
        end_date (datetime.date): The end date of the range.

    Returns:
        meetings (list of date): List of FOMC meeting dates (second day of each meeting).
    """

    meetings = []
    # Add actual meetings within the date range
    for meeting_date in __FOMC_Meetings__:
        if start_date <= meeting_date <= end_date:
            meetings.append(meeting_date)

    years = range(start_date.year, end_date.year + 1)
    for year in years:
        if 2019 <= year <= 2026:
            continue
        for meeting_date in estimate_meetings_for_year(year):
            if start_date <= meeting_date <= end_date:
                meetings.append(meeting_date)
    return sorted(meetings)


def estimate_meetings_for_year(year):
    """
    Estimate the FOMC meeting dates for a given year.

    Parameters:
        year (int): The year for which to estimate meeting dates.

    Returns:
        List[datetime.datetime]: A list of 8 datetime objects representing the estimated meeting dates.
    """
    meetings = []
    # Approximate target days for each meeting based on historical patterns
    meeting_info = [
        (1, 25),   # Late January
        (3, 15),   # Mid-March
        (5, 3),    # Early May
        (6, 14),   # Mid-June
        (7, 26),   # Late July
        (9, 20),   # Mid-September
        (11, 1),   # Early November
        (12, 13),  # Mid-December
    ]
    for month, day in meeting_info:
        # Create the target date
        target_date = dt.datetime(year, month, day)
        # Calculate days to the next Wednesday (weekday 2)
        days_ahead = (2 - target_date.weekday()) % 7
        # Adjust the date to the next Wednesday
        meeting_date = target_date + dt.timedelta(days=days_ahead)
        if not _SIFMA_.is_biz_day(meeting_date):
            meeting_date -= dt.timedelta(days=7)
        meetings.append(meeting_date)
    return meetings


def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Start timer
        start_time = time.time()
        try:
            # Execute the function
            return func(*args, **kwargs)
        finally:
            # End timer and log the elapsed time
            end_time = time.time()
            elapsed_time = end_time - start_time
            logging.info(f"Function {func.__name__} took {elapsed_time:.4f} seconds to execute.")
    return wrapper


if __name__ == '__main__':
    # Check if a date is a business day
    test_date = dt.date(2024, 6, 19)
    print(f"Is {test_date} a SIFMA business day? {_SIFMA_.is_biz_day(test_date)}")
    print(f"Previous business day before {test_date}: {_SIFMA_.prev_biz_day(test_date)}")
    print(f"Next business day after {test_date}: {_SIFMA_.next_biz_day(test_date)}")

    # Example usage
    st = dt.datetime(2018, 4, 1)
    et = dt.datetime(2024, 10, 9)
    days = pd.date_range(st, et, freq='D')
    for day in days:
        if not _SIFMA_.is_biz_day(day):
            if day.dayofweek < 5:
                print(f"{day.date()} is not a SIFMA business day")

    fomc_meetings = generate_fomc_meeting_dates(st, et)
    print("FOMC Meeting Dates (Second Day) between {} and {}:".format(st, et))
    for meeting in fomc_meetings:
        print(meeting.strftime("%B %d, %Y"))

