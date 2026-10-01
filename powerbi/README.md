# Power BI Build Guide

Use the cleaned country-level dataset produced by the Python workflow.

## Model

1. Load the cleaned CSV with **Get Data**.
2. In Power Query, validate date, text and numeric data types.
3. Create a calendar table and relate it to the COVID table on date.
4. Keep cumulative and daily measures conceptually separate.

### Date table

```DAX
DateTable =
CALENDAR(
    MIN(CovidData[date]),
    MAX(CovidData[date])
)
```

Add Year, Month, Month Number and Year-Month columns.

### Core measures

```DAX
New Cases = SUM(CovidData[new_cases])

New Deaths = SUM(CovidData[new_deaths])

Latest Total Cases = MAX(CovidData[total_cases])

Latest Total Deaths = MAX(CovidData[total_deaths])

Case Fatality Rate =
DIVIDE(
    [Latest Total Deaths],
    [Latest Total Cases],
    0
) * 100
```

> Caution: `MAX(total_cases)` is appropriate when the filter context is one country. For multi-country KPI totals, create a latest-row-per-country model rather than summing cumulative values across dates.

## Page 1 — Overview

- KPI cards: cases, deaths, CFR, cases per million, deaths per million
- Location, continent and date slicers
- 7-day case trend
- Top represented countries by cumulative cases

## Page 2 — Cases & Mortality

- 7-day new-case trend
- 7-day new-death trend
- deaths per million by country
- CFR by country
- cases per million vs deaths per million scatter plot

## Page 3 — Testing, Vaccination & Healthcare

- positivity-rate trend
- testing trend
- vaccination coverage
- hospitalization trend
- ICU trend

## Page 4 — Public Health Risk Factors

Use country-level scatter plots:
- median age vs deaths per million
- GDP per capita vs deaths per million
- hospital beds per thousand vs deaths per million
- diabetes prevalence vs deaths per million
- HDI vs deaths per million

Use tooltips for country, population, cases per million and deaths per million.

## Interpretation

Treat relationships as associations. Cross-country comparisons can be affected by reporting quality, surveillance, testing, demographics and timing.
