import datetime as dt


def generate_fomc_meeting_dates(start_date, end_date):
    """
    Generates FOMC meeting dates within the given date range.

    Args:
        start_date (datetime.date): The start date of the range.
        end_date (datetime.date): The end date of the range.

    Returns:
        meetings (list of date): List of FOMC meeting dates (second day of each meeting).
    """
    # Actual FOMC meeting dates from 2019 to 2026
    actual_meetings = [
        # 2019
        dt.date(2019, 1, 30),
        dt.date(2019, 3, 20),
        dt.date(2019, 5, 1),
        dt.date(2019, 6, 19),
        dt.date(2019, 7, 31),
        dt.date(2019, 9, 18),
        dt.date(2019, 10, 30),
        dt.date(2019, 12, 11),
        # 2020
        dt.date(2020, 1, 29),
        # dt.date(2020, 3, 3),  # Emergency meeting
        dt.date(2020, 3, 15),  # Emergency meeting
        dt.date(2020, 4, 29),
        dt.date(2020, 6, 10),
        dt.date(2020, 7, 29),
        dt.date(2020, 9, 16),
        dt.date(2020, 11, 5),
        dt.date(2020, 12, 16),
        # 2021
        dt.date(2021, 1, 27),
        dt.date(2021, 3, 17),
        dt.date(2021, 4, 28),
        dt.date(2021, 6, 16),
        dt.date(2021, 7, 28),
        dt.date(2021, 9, 22),
        dt.date(2021, 11, 3),
        dt.date(2021, 12, 15),
        # 2022
        dt.date(2022, 1, 26),
        dt.date(2022, 3, 16),
        dt.date(2022, 5, 4),
        dt.date(2022, 6, 15),
        dt.date(2022, 7, 27),
        dt.date(2022, 9, 21),
        dt.date(2022, 11, 2),
        dt.date(2022, 12, 14),
        # 2023
        dt.date(2023, 2, 1),
        dt.date(2023, 3, 22),
        dt.date(2023, 5, 3),
        dt.date(2023, 6, 14),
        dt.date(2023, 7, 26),
        dt.date(2023, 9, 20),
        dt.date(2023, 11, 1),
        dt.date(2023, 12, 13),
        # 2024
        dt.date(2024, 1, 31),
        dt.date(2024, 3, 20),
        dt.date(2024, 5, 1),
        dt.date(2024, 6, 12),
        dt.date(2024, 7, 31),
        dt.date(2024, 9, 18),
        dt.date(2024, 10, 30),
        dt.date(2024, 12, 11),
        # 2025
        dt.date(2025, 1, 29),
        dt.date(2025, 3, 19),
        dt.date(2025, 4, 30),
        dt.date(2025, 6, 11),
        dt.date(2025, 7, 30),
        dt.date(2025, 9, 17),
        dt.date(2025, 10, 29),
        dt.date(2025, 12, 10),
        # 2026
        dt.date(2026, 1, 28),
        dt.date(2026, 3, 18),
        dt.date(2026, 4, 29),
        dt.date(2026, 6, 10),
        dt.date(2026, 7, 29),
        dt.date(2026, 9, 16),
        dt.date(2026, 10, 28),
        dt.date(2026, 12, 9),
    ]

    meetings = []

    # Add actual meetings within the date range
    for meeting_date in actual_meetings:
        if start_date <= meeting_date <= end_date:
            meetings.append(meeting_date)

    # Generate estimated meetings outside the range of actual dates
    if start_date.year < 2019 or end_date.year > 2026:
        estimated_meetings = estimate_future_meeting_dates(start_date, end_date)
        meetings.extend(estimated_meetings)

    # Remove duplicates and sort the list
    meetings = sorted(list(set(meetings)))

    return meetings


def estimate_future_meeting_dates(start_date, end_date):
    """
    Estimates FOMC meeting dates using heuristics for years outside 2019-2026.

    Args:
        start_date (datetime.date): The start date of the range.
        end_date (datetime.date): The end date of the range.

    Returns:
        estimated_meetings (list of date): List of estimated meeting dates.
    """
    estimated_meetings = []

    # Determine years that need estimation
    years = set()
    current_year = start_date.year
    while current_year <= end_date.year:
        if current_year < 2019 or current_year > 2026:
            years.add(current_year)
        current_year += 1

    for year in years:
        # Estimate meetings for the year
        meetings = estimate_meetings_for_year(year)
        for meeting_date in meetings:
            if start_date <= meeting_date <= end_date:
                estimated_meetings.append(meeting_date)

    return estimated_meetings


def estimate_meetings_for_year(year):
    """
    Estimates FOMC meeting dates for a given year using heuristics.

    Args:
        year (int): The year for which to estimate meeting dates.

    Returns:
        meetings (list of datetime.date): List of estimated meeting dates.
    """
    meetings = []

    # Typical months when FOMC meetings occur
    meeting_months = [1, 3, 5, 6, 7, 9, 10, 12]

    for month in meeting_months:
        # Find the second day (Wednesday) of the FOMC meeting
        meeting_date = get_meeting_second_day(year, month)
        if meeting_date:
            meetings.append(meeting_date)

    return meetings


def get_meeting_second_day(year, month):
    """
    Finds the second day (Wednesday) of the FOMC meeting for a given month and year.

    Args:
        year (int): The year.
        month (int): The month.

    Returns:
        meeting_date (datetime.date): The second day of the meeting.
    """
    # Get the first day of the month
    first_day = dt.date(year, month, 1)

    # List of potential meeting days (Wednesdays)
    wednesdays = []
    day = first_day
    while day.month == month:
        if day.weekday() == 2:  # 0=Monday, 1=Tuesday, 2=Wednesday
            wednesdays.append(day)
        day += dt.timedelta(days=1)

    if len(wednesdays) >= 3:
        # Use the third Wednesday
        meeting_second_day = wednesdays[2]
    elif len(wednesdays) >= 2:
        # Use the second Wednesday
        meeting_second_day = wednesdays[1]
    elif wednesdays:
        # Use the first Wednesday
        meeting_second_day = wednesdays[0]
    else:
        meeting_second_day = None

    return meeting_second_day


# Example usage:
if __name__ == "__main__":
    start = dt.date(2010, 1, 1)
    end = dt.date(2030, 12, 31)
    fomc_meetings = generate_fomc_meeting_dates(start, end)
    print("FOMC Meeting Dates (Second Day) between {} and {}:".format(start, end))
    for meeting in fomc_meetings:
        print(meeting.strftime("%B %d, %Y"))
