"""
Calculate energy consumption of specified area in Sundsvall and generate related graphs.

This file has been created for the purpose of calculating energy consumption in a small
area in Sundsvall, Sweden as part of a report for LTU's (Luleå's Technical University)
course W0021T - Mekanik och Elkraftteknik.

References:
Available in references.md
"""
import json
import pathlib
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# pylint: disable=import-error
from process import (
    PrivateProperties,
    ElectricVehicles,
    get_csv_data,
    House,
    SawMill,
    SolarPanels,
    StreetLights,
    Tunadal,
)
import generate  # pylint: disable=import-error


def gen_tunadal_data(sun_data: pd.DataFrame) -> Tuple[Tunadal, Tunadal, Tunadal]:
    """Generate data for different time frames in the tunadal area."""

    tunadal_now = Tunadal(
        properties=PrivateProperties(
            house_data=((126.0, 94), (216.0, 13), (330.0, 3)),
            # Properties with district heating 4000 [1]
            # All households in Sundsvall 48137 [4] ansuming that about 20% of the population
            # live in central Sundsvall [2]or apartments and therefore count as less properties
            district_heating_chance=4000.0 / (48137.0 * 0.8),
            sun_data=sun_data,
        ),
        vehicles=ElectricVehicles(n_vehicles=8),
        lights=StreetLights(n_lights=(80, 60, 60), sun_data=sun_data),
        sawmill=SawMill(cubic_meters=650_000),
        solar=SolarPanels(n_panels=(10, 10), sun_data=sun_data),
    )
    tunadal_5_years = Tunadal(
        properties=PrivateProperties(
            house_data=((126.0, 100), (216.0, 16), (330.0, 4)),
            district_heating_chance=0.5,
            sun_data=sun_data,
        ),
        vehicles=ElectricVehicles(n_vehicles=76),
        lights=StreetLights(n_lights=(129, 43, 43), sun_data=sun_data),
        sawmill=SawMill(cubic_meters=900_000.0 * 0.9),  # Assuming 10% more efficient
        solar=SolarPanels(n_panels=(20, 20), sun_data=sun_data),
    )
    tunadal_10_years = Tunadal(
        properties=PrivateProperties(
            house_data=((126.0, 108), (216.0, 19), (330.0, 5)),
            district_heating_chance=0.9,
            sun_data=sun_data,
        ),
        vehicles=ElectricVehicles(n_vehicles=145),
        lights=StreetLights(n_lights=(240, 0, 0), sun_data=sun_data),
        sawmill=SawMill(cubic_meters=1_059_000.0 * 0.8),  # Assuming 20% more efficient
        solar=SolarPanels(n_panels=(33, 3), sun_data=sun_data),
    )
    return tunadal_now, tunadal_5_years, tunadal_10_years


def single_house_figures(sun_data: pd.DataFrame, save: bool = False, prefix: str = ""):
    """Generate figures for individual examples."""
    small = House(size=126.0, sun_data=sun_data, district_heating_chance=0.0, temp=20.0)
    medium = House(size=216.0, sun_data=sun_data, district_heating_chance=0.0, temp=20.0)
    large = House(size=330.0, sun_data=sun_data, district_heating_chance=0.0, temp=20.0)
    temp_diffs_heating = [
        House(
            size=126.0,
            sun_data=sun_data,
            temp=float(temp),
            district_heating_chance=0.0,
        )
        for temp in range(20, 25)
    ]
    generate.fig_differences_house_size(
        small.consumption_by_month,
        medium.consumption_by_month,
        large.consumption_by_month,
        save,
        prefix=prefix,
    )
    generate.fig_temp_diff_stackplot(
        np.array([house.consumption_by_month for house in temp_diffs_heating]),
        save,
        prefix=prefix,
    )
    if save:
        pathlib.Path("data_small_house.json").write_text(
            small.to_json(), encoding="utf-8"
        )
        pathlib.Path("data_medium_house.json").write_text(
            medium.to_json(), encoding="utf-8"
        )
        pathlib.Path("data_large_house.json").write_text(
            large.to_json(), encoding="utf-8"
        )
        pathlib.Path("data_heating_diff_temps.json").write_text(
            json.dumps([i.to_json() for i in temp_diffs_heating], indent=4),
            encoding="utf-8",
        )


def future_figures(
    data: Tuple[Tunadal, Tunadal, Tunadal], save: bool = False, prefix: str = ""
):
    """Generate figures comparing future numbers."""
    generate.fig_plot_compare_now_with_future(
        [
            data[0].consumption_by_week_no_sawmill,
            data[1].consumption_by_week_no_sawmill,
            data[2].consumption_by_week_no_sawmill,
        ],
        save,
        prefix,
    )

    generate.fig_plot_compare_now_with_future(
        [
            data[0].consumption_by_week,
            data[1].consumption_by_week,
            data[2].consumption_by_week,
        ],
        save,
        prefix,
    )


def tunadal_figures(
    data: Tunadal,
    exclude: List[str],
    save: bool = False,
    prefix: str = "",
):
    """Generate data and diagrams for the whole tunadal area."""
    if "daily_avg" not in exclude:
        generate.fig_daily_avg(
            data.properties.consumption_by_day / data.properties.n_houses, save, prefix
        )

    if "daily_total" not in exclude:
        generate.fig_daily_total(data.properties.consumption_by_day, save, prefix)

    if "monthly_total" not in exclude:
        generate.fig_monthly_total(data.properties.consumption_by_month, save, prefix)

    if "over_interval" not in exclude:
        generate.fig_days_over_interval(
            12, data.properties.consumption_by_day, save, prefix
        )
        generate.fig_days_over_interval(
            52, data.properties.consumption_by_day, save, prefix
        )

    if "compare_by_week" not in exclude:
        generate.fig_compare_interval_by_week(
            data.properties.consumption_by_week, save, prefix
        )

    if "compare_by_month" not in exclude:
        generate.fig_compare_interval_by_month(
            data.properties.consumption_by_month, save, prefix
        )

    if "compare_all_but_sawmill_monthly" not in exclude:
        generate.fig_stack_all_but_sawmill_by_interval(
            [
                data.solar.generation_by_month,
                data.properties.consumption_by_month,
                data.vehicles.consumption_by_month,
                data.lights.consumption_by_month,
            ],
            save,
            prefix,
        )

    if "compare_all_month" not in exclude:
        generate.fig_compare_all_data(
            [
                data.properties.consumption_by_week,
                data.vehicles.consumption_by_week,
                data.lights.consumption_by_week,
                data.sawmill.consumption_by_week,
                data.solar.generation_by_week,
            ],
            save,
            prefix,
        )

    if "pie_chart_private_consumption_yearly" not in exclude:
        generate.fig_pie_chart_yearly_cons(
            [
                data.properties.consumption_yearly,
                data.vehicles.consumption_yearly,
                data.lights.consumption_yearly,
            ],
            save,
            prefix,
        )


def main(save: bool):
    """Run all functions."""
    sun_data = get_csv_data()
    data = gen_tunadal_data(sun_data)
    exclude = [
        "daily_avg",
        "daily_total",
        "monthly_total",
        "compare_by_week",
        "compare_by_month",
        "over_interval",
    ]

    single_house_figures(sun_data, save, prefix="[2022] ")
    future_figures(data, save, "[2022 vs 2027 vs 2032] ")
    tunadal_figures(data=data[0], exclude=[], prefix="[2022] ", save=save)
    tunadal_figures(data=data[1], exclude=exclude, prefix="[2027] ", save=save)
    tunadal_figures(data=data[2], exclude=exclude, prefix="[2032] ", save=save)

    pathlib.Path("data_2022.json").write_text(data[0].to_json(), encoding="utf-8")
    pathlib.Path("data_2027.json").write_text(data[1].to_json(), encoding="utf-8")
    pathlib.Path("data_2032.json").write_text(data[2].to_json(), encoding="utf-8")
    plt.show()


if __name__ == "__main__":
    main(save=True)
