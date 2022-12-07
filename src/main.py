"""
Calculate energy consumption of specified area in Sundsvall and generate related graphs.

This file has been created for the purpose of calculating energy consumption in a small
area in Sundsvall, Sweden as part of a report for LTU's (Luleå's Technical University)
course W0021T - Mekanik och Elkraftteknik.

References:
Available in references.md
"""
import matplotlib.pyplot as plt
import numpy as np

from process import Area, House, HouseSize  # pylint: disable=import-error
import generate  # pylint: disable=import-error


def gen_all_figures(save: bool = False):
    """Generate data and diagrams."""
    ex_small = House(HouseSize.SMALL, temp=20.0, district_heating=True)
    ex_medium = House(HouseSize.MEDIUM, temp=20.0, district_heating=True)
    ex_large = House(HouseSize.LARGE, temp=20.0, district_heating=True)
    temp_diffs_district_heating = [
        House(HouseSize.SMALL, temp=float(temp), district_heating=True)
        for temp in range(20, 25)
    ]
    area = Area(94, 13, 3)
    generate.fig_daily_avg(area.avg_energy_consumption_by_day, save)
    generate.fig_monthly_total(area.total_energy_consumption_by_month, save)
    generate.fig_monthly_avg(area.avg_energy_consumption_by_month, save)
    generate.fig_differences_house_size(
        ex_small.energy_consumption_by_month,
        ex_medium.energy_consumption_by_month,
        ex_large.energy_consumption_by_month,
        save,
    )
    generate.fig_differences_house_size(
        ex_small.energy_consumption_by_week,
        ex_medium.energy_consumption_by_week,
        ex_large.energy_consumption_by_week,
        save,
    )
    generate.fig_differences_house_size(
        ex_small.energy_consumption_by_day,
        ex_medium.energy_consumption_by_day,
        ex_large.energy_consumption_by_day,
        save,
    )
    generate.fig_days_over_interval(12, area.avg_energy_consumption_by_day, save)
    generate.fig_days_over_interval(52, area.avg_energy_consumption_by_day, save)
    generate.fig_compare_interval_by_week(area.total_energy_consumption_by_week, save)
    generate.fig_compare_interval_by_month(area.total_energy_consumption_by_month, save)
    generate.fig_temp_diff_stackplot(
        np.array([i.energy_consumption_by_month for i in temp_diffs_district_heating]),
        save,
    )
    plt.show()


if __name__ == "__main__":
    gen_all_figures(save=True)
