"""
Calculate energy consumption of specified area in Sundsvall and generate related graphs.

This file has been created for the purpose of calculating energy consumption in a small
area in Sundsvall, Sweden as part of a report for LTU's (Luleå's Technical University)
course W0021T - Mekanik och Elkraftteknik.

It contains functions and classes to process the incoming data.

References:
Available in references.md
"""
from dataclasses import dataclass, field, InitVar
from enum import Enum
import pathlib
import random
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd


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
    district_heating: InitVar[bool] = False
    energy_consumption_yearly: np.float32 = field(init=False)
    energy_consumption_by_month: npt.NDArray[np.float32] = field(init=False)
    energy_consumption_by_week: npt.NDArray[np.float32] = field(init=False)
    energy_consumption_by_day: npt.NDArray[np.float32] = field(init=False)

    def __post_init__(self, temp, district_heating):
        """Randomize temperature range and type of heating used for each house."""
        # Properties with district heating [1]
        w_dist_heating = 4000.0
        # All households in Sundsvall [4] ansuming that about 20% of the population
        # live in central Sundsvall [2]or apartments and therefore count as less properties
        p_sundsvall = 48137.0 * 0.8
        district_heating = (
            True
            if district_heating
            else bool(random.random() <= (w_dist_heating / p_sundsvall))
        )
        if not temp:
            temp = np.random.uniform(20.0, 25.0)
        self.energy_consumption_yearly = self._energy_consumption_yearly(
            temp,
            district_heating,
        )
        cons_day, cons_week, cons_month = self._energy_consumption_by_time_frame()
        self.energy_consumption_by_day = cons_day
        self.energy_consumption_by_week = cons_week
        self.energy_consumption_by_month = cons_month

    def _energy_consumption_yearly(
        self, temp: np.float32, district_heating: bool
    ) -> np.float32:
        """
        Calculate yearly energy consumption by evaluating size, if district heating
        is used and which temperature the property is set to.
        """
        # District heating saves up to 75% of the consumed energy
        # (in comparison to heating through direct electricity) [3]
        # Formula for energy consumption by size and heating type [5][7]
        yearl_energy_consumption = self.size.value * (40.0 if district_heating else 120.0)
        # Energy consumption can be decreased/incread by 5% per 1°C deviation 21 °C [6]
        percent: np.float32 = 0.05
        temp_limit: np.float32 = 21.0
        variation = yearl_energy_consumption * (percent * (temp - temp_limit))
        return np.float32(yearl_energy_consumption + variation)

    def _energy_consumption_by_time_frame(
        self,
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Calculate energy consumption for each day."""
        sun_data = _get_csv_data()
        energy_consumption_by_day = day_algorithm(
            self.energy_consumption_yearly, sun_data
        )
        days_31 = [0, 2, 4, 6, 7, 9, 11]
        days_30 = [3, 5, 8, 10]
        energy_consumption_by_week = np.empty(shape=(52), dtype=np.float32)
        energy_consumption_by_month = np.empty(shape=(12), dtype=np.float32)
        current_day = 0
        for month in range(12):
            days = 28
            if month in days_31:
                days = 31
            elif month in days_30:
                days = 30
            last_day_of_month = current_day + days

            energy_consumption_by_month[month] = sum(
                energy_consumption_by_day[current_day: current_day + days]
            )

            current_day = last_day_of_month

        for week, start_day in enumerate(range(0, 364, 7)):
            energy_consumption_by_week[week] = sum(
                energy_consumption_by_day[start_day: start_day + 7]
            )
        # Spread value of last day of the year across all weeks as 365 / 7 has a rest of 1
        energy_consumption_by_week += energy_consumption_by_day[364] / 52

        return (
            np.array(energy_consumption_by_day, dtype=np.float32),
            energy_consumption_by_week,
            energy_consumption_by_month,
        )


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


def _get_csv_data():
    """
    Process sunhours csv file. The file contains day, sunrise, sunset,
    if summertime, total hours with sunlight, total hours without sunlight.
    The data within comes from source [8] and has been slightly processed
    (Day of month to day of year).
    """
    data = pd.read_csv(pathlib.Path("sunhours.csv"), usecols=[1, 2, 3, 4, 5])
    data["Daytime"] = np.dot(
        np.array(data.Daytime.str.findall("[0-9]+").tolist()).astype(int), [60, 1]
    )
    data["Nighttime"] = np.dot(
        np.array(data.Nighttime.str.findall("[0-9]+").tolist()).astype(int), [60, 1]
    )
    return data


def day_algorithm(
    yearly_cons: np.float32, sun_data: pd.DataFrame
) -> npt.NDArray[np.float32]:
    """Calculate energy consumption per day from total yearly consumption."""
    total_yearly_nighttime = sun_data["Nighttime"].sum()
    nighttime_per_day = sun_data["Nighttime"].to_numpy(dtype=np.float32)
    modifiers = nighttime_per_day * 10 / total_yearly_nighttime
    vector = np.random.uniform(0.98, 1.02, size=(365)) + modifiers
    normalized_vector = vector / np.linalg.norm(vector)
    # Sum of squared values in a normalized vector is = 1
    normalized_squared = normalized_vector**2 * yearly_cons
    return normalized_squared


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
