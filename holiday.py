import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar, Holiday, nearest_workday,
    USMemorialDay, USLaborDay, USMartinLutherKingJr,
    USPresidentsDay, USThanksgivingDay, GoodFriday, USColumbusDay
)
import datetime


class SIFMAHolidayCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday('NewYearsDay', month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        USMemorialDay,
        Holiday('Juneteenth', month=6, day=19, observance=nearest_workday, start_date='2022-01-01'),
        Holiday('IndependenceDay', month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USColumbusDay,
        Holiday('VeteransDay', month=11, day=11, observance=nearest_workday),
        USThanksgivingDay,
        Holiday('ChristmasDay', month=12, day=25, observance=nearest_workday),
    ]

    # Generate Good Friday dates, excluding those on the first Friday of the month
    good_fridays = GoodFriday.dates(pd.Timestamp('1990-01-01'), pd.Timestamp('2050-12-31'))
    filtered_good_fridays = []
    for gf in good_fridays:
        # Get the first Friday of the month
        first_day = pd.Timestamp(gf.year, gf.month, 1)
        first_friday = first_day + pd.offsets.Week(weekday=4)
        # Include Good Friday if it's not the first Friday of the month
        if gf != first_friday:
            filtered_good_fridays.append(gf)

    # Define special holidays
    special_holidays = [
                           # Hurricane Sandy closure
                           pd.Timestamp('2012-10-30'),
                       ] + filtered_good_fridays


# Define the NYC holiday calendar
class NewYorkHolidayCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday('NewYearsDay', month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        USMemorialDay,
        Holiday('IndependenceDay', month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday('ChristmasDay', month=12, day=25, observance=nearest_workday),
        # Add Juneteenth starting from 2022
        Holiday('Juneteenth', month=6, day=19, observance=nearest_workday, start_date='2022-01-01'),
        # Include additional federal holidays
        Holiday('VeteransDay', month=11, day=11, observance=nearest_workday),
        USColumbusDay,
    ]


# Instantiate the holiday calendars
sifma_calendar = SIFMAHolidayCalendar()
nyc_calendar = NewYorkHolidayCalendar()

# Generate holiday lists
sifma_holidays = sifma_calendar.holidays(start='1990-01-01', end='2050-12-31')
nyc_holidays = nyc_calendar.holidays(start='1990-01-01', end='2050-12-31')

# Combine the holidays and remove duplicates
combined_holidays = pd.to_datetime(sorted(set(sifma_holidays.tolist() + nyc_holidays.tolist())))


class NYTHolidays:
    def __init__(self):
        self.holidays = combined_holidays
        self.holiday_set = set(self.holidays)

    def is_biz_day(self, dt):
        dt = pd.Timestamp(dt)
        return dt.dayofweek < 5 and dt not in self.holiday_set

    def prev_biz_day(self, dt):
        dt = pd.Timestamp(dt) - pd.Timedelta(days=1)
        while not self.is_biz_day(dt):
            dt -= pd.Timedelta(days=1)
        return dt

    def next_biz_day(self, dt):
        dt = pd.Timestamp(dt) + pd.Timedelta(days=1)
        while not self.is_biz_day(dt):
            dt += pd.Timedelta(days=1)
        return dt

    def biz_date_range(self, st, et):
        st = pd.Timestamp(st)
        et = pd.Timestamp(et)
        dates = pd.date_range(start=st, end=et, freq='D')
        biz_days = [dt for dt in dates if self.is_biz_day(dt)]
        return biz_days

    def count_days(self, st, et):
        return len(self.biz_date_range(st, et))


# Instantiate the NYT holidays
NYT = NYTHolidays()

if __name__ == '__main__':
    # Example usage
    st = datetime.datetime(2024, 1, 1)
    et = datetime.datetime(2024, 2, 1)
    print(f"Business days between {st.date()} and {et.date()}: {NYT.count_days(st, et)}")

    # Check if a date is a business day
    test_date = datetime.datetime(2024, 1, 15)
    print(f"Is {test_date.date()} a business day? {NYT.is_biz_day(test_date)}")

    # Get the previous and next business day
    print(f"Previous business day before {test_date.date()}: {NYT.prev_biz_day(test_date).date()}")
    print(f"Next business day after {test_date.date()}: {NYT.next_biz_day(test_date).date()}")

    # Get the list of business days in the range
    biz_days = NYT.biz_date_range(st, et)
    print(f"Business days between {st.date()} and {et.date()}:")
    for day in biz_days:
        print(day.date())
