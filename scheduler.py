import pandas as pd
from dateutil.relativedelta import relativedelta
from holiday import SIFMA  # Import the NYT holidays module


class SOFRSwapScheduler:
    def __init__(
            self,
            ref_date,
            start_date=None,
            tenor='5Y',
            frequency_fixed='12M',  # Annual payments
            frequency_float='12M',  # Annual payments
            day_count_fixed='ACT/360',  # Updated to ACT/360
            day_count_float='ACT/360',
            business_day_convention='Modified Following',
            roll_convention='End of Month',
            holiday_calendar=SIFMA.holiday_set,
    ):
        self.ref_date = pd.Timestamp(ref_date)
        self.holiday_calendar = holiday_calendar

        if start_date is None:
            self.start_date = self.calculate_spot_date()
        else:
            self.start_date = pd.Timestamp(start_date)

        self.tenor = tenor
        self.frequency_fixed = frequency_fixed
        self.frequency_float = frequency_float
        self.day_count_fixed = day_count_fixed
        self.day_count_float = day_count_float
        self.business_day_convention = business_day_convention
        self.roll_convention = roll_convention

        self.end_date = self.calculate_end_date()

        self.fixed_leg_schedule = self.generate_leg_schedule(self.frequency_fixed, self.day_count_fixed)
        self.float_leg_schedule = self.generate_leg_schedule(self.frequency_float, self.day_count_float)

    def calculate_spot_date(self):
        date = self.ref_date
        days_added = 0
        while days_added < 2:
            date += pd.Timedelta(days=1)
            if self.is_business_day(date):
                days_added += 1
        return date

    def calculate_end_date(self):
        num = int(self.tenor[:-1])
        unit = self.tenor[-1].upper()
        if unit == 'Y':
            end_date = self.start_date + relativedelta(years=num)
        elif unit == 'M':
            end_date = self.start_date + relativedelta(months=num)
        else:
            raise ValueError("Unsupported tenor unit. Use 'Y' for years or 'M' for months.")
        return end_date

    def get_third_wednesday(self, year, month):
        first_day = pd.Timestamp(year=year, month=month, day=1)
        first_wednesday = first_day + pd.offsets.Week(weekday=2)
        return first_wednesday + pd.DateOffset(weeks=2)

    def generate_imm_dates(self, start_date, end_date):
        imm_dates = []
        current_year = start_date.year

        while True:
            for month in [3, 6, 9, 12]:
                imm_date = self.get_third_wednesday(current_year, month)
                if imm_date > end_date:
                    return imm_dates  # Exit if the IMM date exceeds the end date
                if imm_date >= start_date:
                    imm_dates.append(imm_date)
            current_year += 1

    def generate_leg_schedule(self, frequency, day_count_convention, payment_delay=2):
        schedule = []
        current_date = self.start_date

        if self.roll_convention == 'IMM':
            imm_dates = self.generate_imm_dates(self.start_date, self.end_date)
            date_list = imm_dates
        else:
            freq_num = int(frequency[:-1])
            freq_unit = frequency[-1].upper()
            date_list = []

            while current_date < self.end_date:
                next_date = current_date
                if freq_unit == 'M':
                    next_date += relativedelta(months=freq_num)
                elif freq_unit == 'Y':
                    next_date += relativedelta(years=freq_num)
                else:
                    raise ValueError("Unsupported frequency unit. Use 'M' for months or 'Y' for years.")

                if self.roll_convention == 'End of Month' and self.is_end_of_month(current_date):
                    next_date = next_date + relativedelta(day=31)

                date_list.append(next_date)
                current_date = next_date

        for i, next_date in enumerate(date_list):
            adj_current_date = self.adjust_date(self.start_date if i == 0 else date_list[i - 1])
            adj_next_date = self.adjust_date(next_date)

            payment_date = adj_next_date
            days_added = 0
            payment_date += pd.Timedelta(days=1)
            while days_added < payment_delay:
                if self.is_business_day(payment_date):
                    days_added += 1
                payment_date += pd.Timedelta(days=1)
            payment_date -= pd.Timedelta(days=1)
            payment_date = self.adjust_date(payment_date)

            dcf = self.calculate_day_count_fraction(adj_current_date, adj_next_date, day_count_convention)

            schedule.append({
                'Accrual Start Date': adj_current_date,
                'Accrual End Date': adj_next_date,
                'Payment Date': payment_date,
                'Day Count Fraction': dcf
            })

        return schedule

    def is_end_of_month(self, date):
        next_month = date + relativedelta(months=1)
        return date.day == (next_month.replace(day=1) - pd.Timedelta(days=1)).day

    def adjust_date(self, date):
        if self.business_day_convention == 'Following':
            adjusted_date = self.following(date)
        elif self.business_day_convention == 'Modified Following':
            adjusted_date = self.modified_following(date)
        elif self.business_day_convention == 'Preceding':
            adjusted_date = self.preceding(date)
        elif self.business_day_convention == 'Modified Preceding':
            adjusted_date = self.modified_preceding(date)
        else:
            adjusted_date = date
        return adjusted_date

    def following(self, date):
        while not self.is_business_day(date):
            date += pd.Timedelta(days=1)
        return date

    def modified_following(self, date):
        original_month = date.month
        date = self.following(date)
        if date.month != original_month:
            date = self.preceding(date)
        return date

    def preceding(self, date):
        while not self.is_business_day(date):
            date -= pd.Timedelta(days=1)
        return date

    def modified_preceding(self, date):
        original_month = date.month
        date = self.preceding(date)
        if date.month != original_month:
            date = self.following(date)
        return date

    def is_business_day(self, date):
        date = pd.Timestamp(date)
        return date.weekday() < 5 and date not in self.holiday_calendar

    def calculate_day_count_fraction(self, start_date, end_date, convention='ACT/360'):
        delta = end_date - start_date
        if convention == 'ACT/360':
            day_count_fraction = delta.days / 360
        else:
            raise ValueError("Unsupported day count convention")
        return day_count_fraction

    def get_fixed_leg_schedule(self):
        return self.fixed_leg_schedule

    def get_float_leg_schedule(self):
        return self.float_leg_schedule

    def get_swap_schedule(self):
        return {
            'Fixed Leg': self.fixed_leg_schedule,
            'Floating Leg': self.float_leg_schedule,
        }


if __name__ == '__main__':
    scheduler = SOFRSwapScheduler(
        ref_date='2024-01-12',  # Trade date
        tenor='5Y',
        day_count_fixed='ACT/360',  # Both legs use ACT/360
        day_count_float='ACT/360',
        business_day_convention='Modified Following',
        roll_convention='End of Month',
    )

    fixed_schedule = scheduler.get_fixed_leg_schedule()
    float_schedule = scheduler.get_float_leg_schedule()

    print("Fixed Leg Schedule:")
    print(f"{'Period':<6} {'Accrual Start':<15} {'Accrual End':<15} {'Payment Date':<15} {'DCF':<10}")
    for i, period in enumerate(fixed_schedule, 1):
        print(
            f"{i:<6} {period['Accrual Start Date'].strftime('%Y-%m-%d'):<15} {period['Accrual End Date'].strftime('%Y-%m-%d'):<15} {period['Payment Date'].strftime('%Y-%m-%d'):<15} {period['Day Count Fraction']:<10.6f}")

    print("\nFloating Leg Schedule:")
    print(f"{'Period':<6} {'Accrual Start':<15} {'Accrual End':<15} {'Payment Date':<15} {'DCF':<10}")
    for i, period in enumerate(float_schedule, 1):
        print(
            f"{i:<6} {period['Accrual Start Date'].strftime('%Y-%m-%d'):<15} {period['Accrual End Date'].strftime('%Y-%m-%d'):<15} {period['Payment Date'].strftime('%Y-%m-%d'):<15} {period['Day Count Fraction']:<10.6f}")
