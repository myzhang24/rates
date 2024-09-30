import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar, Holiday, nearest_workday, next_workday,
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
        # Include Good Friday if it's not the first Friday of the month (NFP publishing days)
        if gf != first_friday:
            filtered_good_fridays.append(gf)

    # Define special holidays
    special_holidays = [
                           # Hurricane Sandy closure
                           pd.Timestamp('2012-10-30'),
                       ] + filtered_good_fridays


class NYFedHolidayCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday('NewYearsDay', month=1, day=1, observance=next_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        USMemorialDay,
        Holiday('Juneteenth', month=6, day=19, observance=next_workday, start_date='2022-01-01'),
        Holiday('IndependenceDay', month=7, day=4, observance=next_workday),
        USLaborDay,
        USColumbusDay,
        Holiday('VeteransDay', month=11, day=11, observance=next_workday),
        USThanksgivingDay,
        Holiday('ChristmasDay', month=12, day=25, observance=next_workday),
    ]


sifma_holidays = SIFMAHolidayCalendar().holidays(start='1990-01-01', end='2050-12-31')
nyfed_holidays = NYFedHolidayCalendar().holidays(start='1990-01-01', end='2050-12-31')


class Holidays:
    def __init__(self, calendar_class):
        self.holidays = calendar_class
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


# Instantiate the SIFMA and NYFED holidays
SIFMA = Holidays(sifma_holidays)
NYFED = Holidays(nyfed_holidays)

if __name__ == '__main__':
    # Check if a date is a business day
    test_date = datetime.datetime(2024, 1, 15)
    print(f"Is {test_date.date()} a business day? {SIFMA.is_biz_day(test_date)}")
    print(f"Previous business day before {test_date.date()}: {SIFMA.prev_biz_day(test_date).date()}")
    print(f"Next business day after {test_date.date()}: {SIFMA.next_biz_day(test_date).date()}")

    # Example usage
    st = datetime.datetime(2024, 1, 1)
    et = datetime.datetime(2024, 2, 1)
    biz_days = SIFMA.biz_date_range(st, et)
    print(f"Business days between {st.date()} and {et.date()}:")
    for day in biz_days:
        print(day.date())
