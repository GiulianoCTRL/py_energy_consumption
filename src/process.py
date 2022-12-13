"""
Calculate energy consumption of specified area in Sundsvall and generate related graphs.

This file has been created for the purpose of calculating energy consumption in a small
area in Sundsvall, Sweden as part of a report for LTU's (Luleå's Technical University)
course W0021T - Mekanik och Elkraftteknik.

It contains functions and classes to process the incoming data.

References:
Available in references.md
"""
from dataclasses import asdict, dataclass, field, InitVar
import json
from math import sqrt
import pathlib
import random
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy data."""

    def default(self, o):
        """Default encoder for NumpyEncoder."""

        int_type = (
            np.int_,
            np.intc,
            np.intp,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        )
        float_type = (np.float_, np.float16, np.float32, np.float64)

        if isinstance(o, int_type):
            return int(o)

        if isinstance(o, float_type):
            return float(o)

        if isinstance(o, np.ndarray):
            return o.tolist()

        return json.JSONEncoder.default(self, o)


@dataclass
class ElectricVehicles:
    """Estimate power consumption of for electric vehicles."""

    n_vehicles: InitVar[int]
    consumption_by_day: npt.NDArray[np.float32] = field(init=False)
    consumption_by_week: npt.NDArray[np.float32] = field(init=False)
    consumption_by_month: npt.NDArray[np.float32] = field(init=False)
    consumption_yearly: np.float32 = field(init=False)

    def __post_init__(self, n_vehicles: int):
        """Estimate electric vehicles power draw (2kWh / 10km)."""
        # A car drives between 10 and 50 km each day.
        vehicle_daily_distances = np.random.uniform(1.0, 5.0, size=(n_vehicles))
        consumption_by_day = np.zeros(shape=(365), dtype=np.float32)
        for distance in vehicle_daily_distances:
            consumption_by_day += np.ones(shape=(365), dtype=np.float32) * 2.0 * distance
        self.consumption_by_day = consumption_by_day
        (self.consumption_by_week, self.consumption_by_month) = energy_by_interval(
            self.consumption_by_day
        )
        self.consumption_yearly = self.consumption_by_day.sum()


@dataclass
class SawMill:
    """Estimate SawMills power consumption."""

    cubic_meters: InitVar[float]
    consumption_by_day: npt.NDArray[np.float32] = field(init=False)
    consumption_by_week: npt.NDArray[np.float32] = field(init=False)
    consumption_by_month: npt.NDArray[np.float32] = field(init=False)
    consumption_yearly: np.float32 = field(init=False)

    def __post_init__(self, cubic_meters: float):
        """Estimate electric vehicles power draw (2kWh / 10km)."""
        cons_day = 144.0 * cubic_meters / 365
        # type: ignore
        self.consumption_by_day = np.ones(shape=(365), dtype=np.float32) * cons_day
        (self.consumption_by_week, self.consumption_by_month) = energy_by_interval(
            self.consumption_by_day
        )
        self.consumption_yearly = self.consumption_by_day.sum()


@dataclass
class StreetLights:
    """
    Estimate power consumption of LED, natrium and mercury street lights.

    :param n_lights:    Tuple(amount LED, amount Natrium, amount mercury)
    """

    n_lights: InitVar[Tuple[int, int, int]]
    sun_data: InitVar[pd.DataFrame]
    consumption_by_day: npt.NDArray[np.float32] = field(init=False)
    consumption_by_week: npt.NDArray[np.float32] = field(init=False)
    consumption_by_month: npt.NDArray[np.float32] = field(init=False)
    consumption_yearly: np.float32 = field(init=False)

    def __post_init__(self, n_lights: Tuple[int, int, int], sun_data: pd.DataFrame):
        """
        Estimate street light power consumption.

        :param n_lights:    Tuple(amount LED, amount Natrium, amount mercury)
        """
        night_hours_by_day = sun_data["Nighttime"].to_numpy(dtype=np.float32) / 60.0
        cons_lamps: Tuple[float, float, float] = (
            0.05 * n_lights[0],
            0.25 * n_lights[1],
            1.00 * n_lights[2],
        )  # type: ignore
        self.consumption_by_day = night_hours_by_day * sum(cons_lamps)  # type: ignore
        (
            self.consumption_by_week,
            self.consumption_by_month,
        ) = energy_by_interval(self.consumption_by_day)
        self.consumption_yearly = self.consumption_by_day.sum()


@dataclass
class SolarPanels:
    """
    Power generation and consumption for solar panels, electric vehicles and
    street lights.

    :param n_panels:        Tuple(amount 5kW panels, amount 15kW panels)
    """

    n_panels: InitVar[Tuple[int, int]]
    sun_data: InitVar[pd.DataFrame]
    generation_by_day: npt.NDArray[np.float32] = field(init=False)
    generation_by_week: npt.NDArray[np.float32] = field(init=False)
    generation_by_month: npt.NDArray[np.float32] = field(init=False)
    generation_yearly: np.float32 = field(init=False)

    def __post_init__(
        self,
        n_panels: Tuple[int, int],
        sun_data: pd.DataFrame,
    ):
        """
        Estimate solar power production.

        :param n_panels:        Tuple(amount 5kW panels, amount 15kW panels)
        """
        sun_hours_by_day: npt.NDArray[np.float32] = (
            sun_data["Daytime"].to_numpy(dtype=np.float32) / 60.0
        )
        # 20 % effectivness
        self.generation_by_day = (
            sun_hours_by_day * 0.20 * (5.0 * n_panels[0] + 15.0 * n_panels[1])
        )
        (
            self.generation_by_week,
            self.generation_by_month,
        ) = energy_by_interval(self.generation_by_day)
        self.generation_yearly = self.generation_by_day.sum()


@dataclass
class House:
    """
    General information needed to calculate energy consumption for each individuell
    house. All energy consumbtion values are represented in kWh.
    """

    size: InitVar[float]
    sun_data: InitVar[pd.DataFrame]
    district_heating_chance: InitVar[float]
    temp: InitVar[np.float32] = 0.0
    consumption_yearly: np.float32 = field(init=False)
    consumption_by_month: npt.NDArray[np.float32] = field(init=False)
    consumption_by_week: npt.NDArray[np.float32] = field(init=False)
    consumption_by_day: npt.NDArray[np.float32] = field(init=False)

    def __post_init__(
        self,
        size: np.float32,
        sun_data: pd.DataFrame,
        district_heating_chance: float,
        temp: np.float32,
    ):
        """Randomize temperature range and type of heating used for each house."""
        district_heating = bool(random.random() < district_heating_chance)
        if not temp and not district_heating:
            temp = np.random.uniform(20.0, 25.0)  # type: ignore
        # If district heating room temperature is not affecting consumption
        elif not temp and district_heating:
            temp = np.float32(21.0)

        self.consumption_yearly = self._energy_consumption_yearly(
            size,
            temp,
            district_heating,
        )
        self.consumption_by_day = day_algorithm(
            self.consumption_yearly, sun_data, heating=district_heating
        )
        (
            self.consumption_by_week,
            self.consumption_by_month,
        ) = energy_by_interval(self.consumption_by_day)

    def _energy_consumption_yearly(
        self, size: np.float32, temp: np.float32, district_heating: bool
    ) -> np.float32:
        """
        Calculate yearly energy consumption by evaluating size, if district heating
        is used and which temperature the property is set to.
        """
        # District heating saves up to 75% of the consumed energy
        # (in comparison to heating through direct electricity) [3]
        # Formula for energy consumption by size and heating type [5][7]
        yearl_energy_consumption = size * (40.0 if district_heating else 120.0)
        # Energy consumption can be decreased/incread by 5% per 1°C deviation 21 °C [6]
        variation = yearl_energy_consumption * (0.05 * (temp - 21.0))  # type: ignore
        return np.float32(yearl_energy_consumption + variation)


@dataclass
class PrivateProperties:
    """
    Energy consumption for specific private properties.

    :param houses:      Tuple((size, amount), (size, amount), (size, amount))

    """

    house_data: InitVar[Tuple[Tuple[float, int], Tuple[float, int], Tuple[float, int]]]
    district_heating_chance: InitVar[float]
    sun_data: InitVar[pd.DataFrame]
    consumption_yearly: np.float32 = field(init=False)
    consumption_by_month: npt.NDArray[np.float32] = field(init=False)
    consumption_by_week: npt.NDArray[np.float32] = field(init=False)
    consumption_by_day: npt.NDArray[np.float32] = field(init=False)
    peak_consumption: np.float32 = field(init=False)
    n_houses: int = field(init=False)

    def __post_init__(
        self,
        house_data: Tuple[Tuple[float, int], Tuple[float, int], Tuple[float, int]],
        district_heating_chance: float,
        sun_data: pd.DataFrame,
    ):
        """Calculate total and averages for all houses in the area."""
        houses: List[House] = generate_houses(
            sun_data=sun_data,
            house_data=house_data,
            district_heating_chance=district_heating_chance,
        )
        self.n_houses = len(houses)
        self.consumption_yearly = np.float32(
            sum(house.consumption_yearly for house in houses)
        )
        self.consumption_by_month = aggregate_consumption_lists(12, houses)
        self.consumption_by_week = aggregate_consumption_lists(52, houses)
        self.consumption_by_day = aggregate_consumption_lists(365, houses)

        # Velanders konstant category EL350 [9]
        vel_el350 = (0.000314, 0.086054)
        self.peak_consumption = vel_el350[0] * self.consumption_yearly + vel_el350[
            1
        ] * sqrt(self.consumption_yearly)


@dataclass
class Tunadal:
    """Aggregate data for tunadals area."""

    properties: PrivateProperties
    vehicles: ElectricVehicles
    lights: StreetLights
    sawmill: SawMill
    solar: SolarPanels
    consumption_yearly: np.float32 = field(init=False)
    consumption_by_month: npt.NDArray[np.float32] = field(init=False)
    consumption_by_week: npt.NDArray[np.float32] = field(init=False)
    consumption_by_day: npt.NDArray[np.float32] = field(init=False)
    consumption_yearly_no_sawmill: np.float32 = field(init=False)
    consumption_by_month_no_sawmill: npt.NDArray[np.float32] = field(init=False)
    consumption_by_week_no_sawmill: npt.NDArray[np.float32] = field(init=False)
    consumption_by_day_no_sawmill: npt.NDArray[np.float32] = field(init=False)

    def __post_init__(self):
        """Calculate aggregated values for tunadals area."""
        self.consumption_by_day_no_sawmill = (
            self.properties.consumption_by_day
            + self.vehicles.consumption_by_day
            + self.lights.consumption_by_day
            - self.solar.generation_by_day
        )
        (
            self.consumption_by_week_no_sawmill,
            self.consumption_by_month_no_sawmill,
        ) = energy_by_interval(self.consumption_by_day_no_sawmill)
        self.consumption_yearly_no_sawmill = self.consumption_by_day_no_sawmill.sum()
        self.consumption_by_day = (
            self.consumption_by_day_no_sawmill + self.sawmill.consumption_by_day
        )
        self.consumption_by_week = (
            self.consumption_by_week_no_sawmill + self.sawmill.consumption_by_week
        )
        self.consumption_by_month = (
            self.consumption_by_month_no_sawmill + self.sawmill.consumption_by_month
        )
        self.consumption_yearly = (
            self.consumption_yearly_no_sawmill + self.sawmill.consumption_yearly
        )

    def to_dict(self):
        """Represent data as dict."""
        return asdict(self)

    def to_json(self):
        """Represent data as json."""
        return json.dumps(self.to_dict(), indent=4, cls=NumpyEncoder)


def get_csv_data():
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
    yearly_cons: np.float32, sun_data: pd.DataFrame, heating: bool = True
) -> npt.NDArray[np.float32]:
    """Calculate energy consumption per day from total yearly consumption."""
    total_yearly_nighttime = sun_data["Nighttime"].sum()
    nighttime_per_day = sun_data["Nighttime"].to_numpy(dtype=np.float32)

    # Fluctuations in temperature affect households with district heating less
    heat_modifier = 100 if heating else 50
    modifiers = nighttime_per_day * heat_modifier / total_yearly_nighttime
    vector = np.random.uniform(0.98, 1.02, size=(365)) + modifiers
    normalized_vector = vector / np.linalg.norm(vector)
    # Sum of squared values in a normalized vector is = 1
    normalized_squared = normalized_vector**2 * yearly_cons
    return normalized_squared


def generate_houses(
    sun_data: pd.DataFrame,
    house_data: Tuple[Tuple[float, int], Tuple[float, int], Tuple[float, int]],
    district_heating_chance: float,
) -> List[House]:
    """Generate houses for the dataset."""
    house_list = []
    for (size, amount) in house_data:
        house_list += [
            House(
                size=size,
                sun_data=sun_data,
                district_heating_chance=district_heating_chance,
            )
            for _ in range(amount)
        ]
    return house_list


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
                aggregated += house.consumption_by_month[interval]
            elif target_length == 52:
                aggregated += house.consumption_by_week[interval]
            else:
                aggregated += house.consumption_by_day[interval]
        aggregated_values.append(aggregated)

    return np.array(aggregated_values, dtype=np.float32)


def energy_by_interval(
    energy_by_day: npt.NDArray[np.float32],
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """
    Calculate energy consumption/generation for weekly and monthly intervals by splitting
    it into the respective time interval.
    ."""
    days_31 = [0, 2, 4, 6, 7, 9, 11]
    days_30 = [3, 5, 8, 10]
    energy_by_week = np.empty(shape=(52), dtype=np.float32)
    energy_by_month = np.empty(shape=(12), dtype=np.float32)
    current_day = 0
    for month in range(12):
        days = 28
        if month in days_31:
            days = 31
        elif month in days_30:
            days = 30
        last_day_of_month = current_day + days

        energy_by_month[month] = sum(energy_by_day[current_day : current_day + days])

        current_day = last_day_of_month

    for week, start_day in enumerate(range(0, 364, 7)):
        energy_by_week[week] = sum(energy_by_day[start_day : start_day + 7])
    # Spread value of last day of the year across all weeks as 365 / 7 has a rest of 1
    energy_by_week += energy_by_day[364] / 52

    return (
        energy_by_week,
        energy_by_month,
    )
