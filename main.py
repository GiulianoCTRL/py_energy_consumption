"""
Calculate energy consumption of specified area in Sundsvall and generate related graphs.

This file has been created for the purpose of calculating energy consumption in a small
area in Sundsvall, Sweden as part of a report for LTU's (Luleå's Technical University)
course W0021T - Mekanik och Elkraftteknik.

References:
Available in references.md

Copyright © 2022 Giuliano Ruggeri
"""
# pylint: disable=no-member
from dataclasses import dataclass, field, InitVar
from enum import Enum
from math import sqrt
import random
from typing import List, Union

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


# region:           Classes & Enums
class HouseSize(Enum):
    """House size in square meters for the average house in it's size class."""

    SMALL = 126.0
    MEDIUM = 216.0
    LARGE = 330.0


@dataclass
class House:
    """
    General information needed to calculate energy consumption for each individuell
    house. All energy consumbtion values are represented in kWh.
    """

    size: HouseSize
    temp: InitVar[np.float32] = 0.0
    energy_consumption_yearly: np.float32 = field(init=False)
    energy_consumption_by_month: npt.NDArray[np.float32] = field(init=False)
    energy_consumption_by_week: npt.NDArray[np.float32] = field(init=False)
    energy_consumption_by_day: npt.NDArray[np.float32] = field(init=False)

    def __post_init__(self, temp):
        """Randomize temperature range and type of heating used for each house."""
        # Properties with district heating [1]
        w_dist_heating = 4000.0
        # All households in Sundsvall [4] ansuming that about 20% of the population
        # live in central Sundsvall or apartments and therefore count as less properties
        p_sundsvall = 48137.0 * 0.8
        district_heating = bool(random.random() <= (w_dist_heating / p_sundsvall))
        if not temp:
            temp = np.random.uniform(20.0, 25.0)
        self.energy_consumption_yearly = self._energy_consumption_yearly(
            temp,
            district_heating,
        )
        self.energy_consumption_by_month = self._energy_consumption_by_month()
        self.energy_consumption_by_day = self._energy_consumption_by_day()
        self.energy_consumption_by_week = self._energy_consumption_by_week()

    def _energy_consumption_yearly(
        self, temp: np.float32, district_heating: bool
    ) -> np.float32:
        """
        Calculate yearly energy consumption by evaluating size, if district heating
        is used and which temperature the property is set to.
        """
        # District heating saves up to 75% of the consumed energy
        # (in comparison to heating through direct electricity) [3]
        # Formula for energy consumption by size and heating type [5]
        yearl_energy_consumption = self.size.value * (40.0 if district_heating else 120.0)
        # Energy consumption can be decreased/incread by 5% per 1°C deviation 21 °C [6]
        yearl_energy_consumption += yearl_energy_consumption * (0.05 * (temp - 21.0)) # type: ignore
        return np.float32(yearl_energy_consumption)

    def _energy_consumption_by_month(self) -> npt.NDArray[np.float32]:
        """Calculate energy consumption for each month."""
        # Percentage of yearly consumption for each month [7]
        consumption_month = [
            0.14,  # January
            0.12,  # February
            0.12,  # March
            0.10,  # April
            0.06,  # May
            0.04,  # June
            0.03,  # July
            0.03,  # August
            0.05,  # September
            0.08,  # October
            0.10,  # November
            0.13,  # December
        ]
        return np.array(
            [self.energy_consumption_yearly * percent for percent in consumption_month],
            dtype=np.float32,
        )

    def _energy_consumption_by_day(self) -> npt.NDArray[np.float32]:
        """Calculate energy consumption for each day."""
        days_31 = [0, 2, 4, 6, 7, 9, 11]
        days_30 = [3, 5, 8, 10]
        energy_consumption_by_day = np.empty(shape=(365), dtype=np.float32)
        current_day = 0
        for month, monthly_cons in enumerate(self.energy_consumption_by_month):
            days = 28
            if month in days_31:
                days = 31
            elif month in days_30:
                days = 30
            monthly_consumption_by_day = random_vector_with_specific_sum(
                1.0, 1.1, days, monthly_cons
            )

            energy_consumption_by_day[
                current_day : current_day + days
            ] = monthly_consumption_by_day
            current_day += days
        return energy_consumption_by_day

    def _energy_consumption_by_week(self) -> npt.NDArray[np.float32]:
        """Calculate weekly energy consumption by summing daily values each week."""
        # Return every seventh number between 0 and 364
        consumption_by_week = np.empty(shape=(52), dtype=np.float32)
        for index, day in enumerate(range(0, 364, 7)):
            weekly_consumption = sum(self.energy_consumption_by_day[day : day + 7])
            # The value of the leftover day will be added to the last week
            if day == 357:
                weekly_consumption += self.energy_consumption_by_day[364]
            consumption_by_week[index] = weekly_consumption
        return consumption_by_week


# pylint: disable=too-many-instance-attributes
@dataclass
class Area:
    """Energy consumption for specific Area."""

    n_small_houses: InitVar[int]
    n_medium_houses: InitVar[int]
    n_large_houses: InitVar[int]
    total_energy_consumption_yearly: np.float32 = field(init=False)
    total_energy_consumption_by_month: npt.NDArray[np.float32] = field(init=False)
    total_energy_consumption_by_week: npt.NDArray[np.float32] = field(init=False)
    total_energy_consumption_by_day: npt.NDArray[np.float32] = field(init=False)
    avg_energy_consumption_yearly: np.float32 = field(init=False)
    avg_energy_consumption_by_month: npt.NDArray[np.float32] = field(init=False)
    avg_energy_consumption_by_week: npt.NDArray[np.float32] = field(init=False)
    avg_energy_consumption_by_day: npt.NDArray[np.float32] = field(init=False)

    def __post_init__(
        self, n_small_houses: int, n_medium_houses: int, n_large_houses: int
    ):
        """Calculate total and averages for all houses in the area."""
        houses = generate_houses(n_small_houses, n_medium_houses, n_large_houses)
        self.total_energy_consumption_yearly = np.float32(
            sum(house.energy_consumption_yearly for house in houses)
        )
        self.total_energy_consumption_by_month = aggregate_consumption_lists(12, houses)
        self.total_energy_consumption_by_week = aggregate_consumption_lists(52, houses)
        self.total_energy_consumption_by_day = aggregate_consumption_lists(365, houses)
        self.avg_energy_consumption_yearly = np.float32(
            self.total_energy_consumption_yearly / len(houses)
        )
        self.avg_energy_consumption_by_month = np.array(
            [i / len(houses) for i in self.total_energy_consumption_by_month],
            dtype=np.float32,
        )

        self.avg_energy_consumption_by_week = np.array(
            [i / len(houses) for i in self.total_energy_consumption_by_week],
            dtype=np.float32,
        )

        self.avg_energy_consumption_by_day = np.array(
            [i / len(houses) for i in self.total_energy_consumption_by_day],
            dtype=np.float32,
        )


# endregion:                Classes & Enums


# region:           Functions
def random_vector_with_specific_sum(
    start: Union[np.float32, float],
    stop: Union[np.float32, float],
    n_items: int,
    specific_sum: np.float32,
) -> npt.NDArray[np.float32]:
    """Generate a random vector whose values add up to a specified sum."""
    vector = [np.random.uniform(start, stop) for _ in range(n_items)]
    normalized_vector = [value / sqrt(sum(i**2 for i in vector)) for value in vector]
    # Sum of squared values in a normalized vector is = 1
    return np.array(
        [value**2 * specific_sum for value in normalized_vector], dtype=np.float32
    )


def generate_houses(small: int, medium: int, large: int) -> List[House]:
    """Generate houses for the dataset."""
    small_houses = [House(HouseSize.SMALL) for _ in range(small)]
    medium_houses = [House(HouseSize.MEDIUM) for _ in range(medium)]
    large_houses = [House(HouseSize.LARGE) for _ in range(large)]
    return small_houses + medium_houses + large_houses


def aggregate_consumption_lists(
    target_length: int, house_list: List[House]
) -> npt.NDArray[np.float32]:
    """
    Aggregate values in list and either take the average or total.
    """
    assert target_length in (12, 52, 365)
    aggregated_values = []
    for interval in range(target_length):
        aggregated = 0.0
        for house in house_list:
            if target_length == 12:
                aggregated += house.energy_consumption_by_month[interval]
            elif target_length == 52:
                aggregated += house.energy_consumption_by_week[interval]
            else:
                aggregated += house.energy_consumption_by_day[interval]
        aggregated_values.append(aggregated)

    return np.array(aggregated_values, dtype=np.float32)


# pylint: disable=invalid-name
def gen_figure_daily_avg(data: npt.NDArray[np.float32]):
    """Generate figure with daily averages."""
    _, ax = plt.subplots()
    x = np.arange(365, dtype=np.int32)
    ax.set_title("Energiförbrukning per dygn")
    ax.set_ylabel("kWh")
    ax.set_xlabel("Dygn")

    sizes = data / (np.ones(shape=(len(data))) * 10) * np.random.uniform(0.95, 1.05)
    colors = np.random.uniform(15, 80, len(data))
    ax.scatter(x, data, sizes=sizes, c=colors)
    ax.plot(x, data, linewidth=1, color="blue", label="medelvärde")


# pylint: disable=invalid-name
def gen_figure_monthly_avg_vs_total(
    data_avg: npt.NDArray[np.float32], data_tot: npt.NDArray[np.float32]
):
    """Generate monthly average vs total."""

    x = np.arange(12, dtype=np.int32)
    _, ax = plt.subplots()

    ax.set_title("Elförbrukning total och medelvärde per månad")
    ax.set_xlabel("Månad")
    ax.set_ylabel("kWh")

    ax.fill_between(x, data_avg, data_tot, alpha=0.5, linewidth=0)
    ax.set_yscale("log")
    ax.plot(x, data_tot, linewidth=2, color="red", label="total")
    ax.plot(x, data_avg, linewidth=2, color="blue", label="medelvärde")
    ax.plot(x, data_tot / data_avg, linewidth=2, color="green", label="kvotient")
    ax.legend()


# pylint: disable=invalid-name
def gen_figure_period_by_days(data: npt.NDArray[np.float32]):
    """Generate daily consumption averaged weekly and monthly."""
    weeks_by_days = np.empty(shape=(52, 7))
    months_by_days = np.empty(shape=(12, 30))
    # Get a 52 x 7 matrix
    for index, start_day in enumerate(range(0, 364, 7)):
        weeks_by_days[index] = data[start_day : start_day + 7]

    # Get a 12 x 30 matrix
    for index, start_day in enumerate(range(0, 330, 30)):
        months_by_days[index] = data[start_day : start_day + 30]

    fig_weeks, ax_weeks = plt.subplots()
    ax_weeks.set_xlabel("dygn")
    ax_weeks.set_ylabel("vecka")
    ax_weeks.set_title("Dagsförbrukning alla veckor")

    im_weeks = ax_weeks.pcolormesh(weeks_by_days)
    fig_weeks.colorbar(im_weeks, ax=ax_weeks, label="kWh")
    ax_weeks.legend()

    fig_months, ax_months = plt.subplots()
    ax_months.set_xlabel("dygn")
    ax_months.set_ylabel("månad")
    ax_months.set_title("Dagsförbrukning alla månader")

    im = ax_months.pcolormesh(months_by_days)
    fig_months.colorbar(im, ax=ax_months, label="kWh")
    ax_months.legend()


# pylint: disable=invalid-name
def gen_figure_differences_house_size(
    data_small: npt.NDArray[np.float32],
    data_medium: npt.NDArray[np.float32],
    data_large: npt.NDArray[np.float32],
):
    """Generate graph with differences in house size."""
    time_period = ""
    if (length := len(data_small)) == 12:
        time_period = "månad"
    elif length == 52:
        time_period = "vecka"
    elif length == 365:
        time_period = "dygn"

    x = np.arange(len(data_small), dtype=np.int32)
    _, ax = plt.subplots()

    ax.set_title(
        f"Elförbrukning - skillnader i hus storlek (Samma temperatur) per {time_period}"
    )
    ax.set_xlabel(time_period)
    ax.set_ylabel("kWh")

    ax.fill_between(x, data_small, data_large, alpha=0.5, linewidth=0)
    ax.plot(x, data_small, linewidth=2, color="blue", label="små hus")
    ax.plot(x, data_medium, linewidth=2, color="green", label="medium hus")
    ax.plot(x, data_large, linewidth=2, color="red", label="stor hus")
    ax.legend()


def gen_all_figures():
    """Generate data and diagrams."""
    ex_small = House(HouseSize.SMALL, temp=20.0)
    ex_medium = House(HouseSize.MEDIUM, temp=20.0)
    ex_large = House(HouseSize.LARGE, temp=20.0)
    area = Area(94, 13, 3)
    gen_figure_daily_avg(area.avg_energy_consumption_by_day)
    gen_figure_monthly_avg_vs_total(
        area.avg_energy_consumption_by_month, area.total_energy_consumption_by_month
    )
    gen_figure_differences_house_size(
        ex_small.energy_consumption_by_month,
        ex_medium.energy_consumption_by_month,
        ex_large.energy_consumption_by_month,
    )
    gen_figure_differences_house_size(
        ex_small.energy_consumption_by_week,
        ex_medium.energy_consumption_by_week,
        ex_large.energy_consumption_by_week,
    )
    gen_figure_differences_house_size(
        ex_small.energy_consumption_by_day,
        ex_medium.energy_consumption_by_day,
        ex_large.energy_consumption_by_day,
    )
    gen_figure_period_by_days(area.avg_energy_consumption_by_day)
    plt.show()


# endregion:        Functions

if __name__ == "__main__":
    gen_all_figures()
