# COVID-19 Public Health Analytics

An end-to-end analytics portfolio project examining COVID-19 cases, deaths, population-adjusted burden, testing, vaccination, healthcare utilization, and public-health risk factors using **Python, SQL, Power BI, and Tableau**.

## Project objective

The project is organized around one question:

> How did COVID-19 cases, mortality, testing, vaccination, and healthcare burden evolve over time and across locations, and which demographic, socioeconomic, and healthcare factors were associated with differences in outcomes?

The workflow demonstrates data quality assessment, exploratory analysis, SQL querying, time-series analysis, population-adjusted comparisons, dashboard design, and responsible interpretation of observational public-health data.

## Tools

- **Python:** pandas, NumPy, Matplotlib, statistical exploration
- **SQL:** data-quality checks, aggregations, CTEs, ranking and time-series queries
- **Power BI:** KPI reporting, DAX measures, interactive trend and risk-factor dashboards
- **Tableau:** country comparisons, trends, scatter plots and dashboard storytelling

## Analytical questions

1. What time period and locations are represented?
2. Are there missing, duplicate, negative, or potentially corrected observations?
3. Which represented countries have the highest cumulative cases and deaths?
4. How do rankings change when cases and deaths are adjusted for population?
5. How did new cases and deaths change over time?
6. When did major case and mortality peaks occur?
7. How do 7-day averages clarify pandemic waves?
8. How did testing volume and positivity change over time?
9. How did vaccination coverage change?
10. How were vaccination coverage and mortality associated?
11. How did hospitalization and ICU burden change during case surges?
12. How were median age, GDP per capita, diabetes prevalence, hospital-bed capacity and HDI associated with mortality?

## Repository structure

```text
.
├── README.md
├── Covid_19_Dataset_Analysis (1).ipynb
├── Covid_19_ Analysis.py
├── Machine Learning analysis using Covid 19 Dataset.py
├── Covid 19_visual country risk dashboard.py
├── covid_19_cleaned.csv
├── country_ml_risk_report.csv
├── sql/
│   └── covid19_analysis.sql
├── powerbi/
│   └── README.md
├── tableau/
│   └── README.md
└── existing analysis images
```

## Python workflow

The Python analysis follows this sequence:

1. Load and inspect the data.
2. Parse dates and inspect coverage.
3. Audit missing values and duplicates.
4. Check negative daily values and reporting corrections.
5. Separate countries from regional/world aggregates where appropriate.
6. Validate derived measures such as case-fatality rate.
7. Build latest-country snapshots for cumulative comparisons.
8. Analyze cases and deaths over time using 7-day averages.
9. Compare cases/deaths per million rather than relying only on raw totals.
10. Examine testing, vaccination, hospitalization and ICU indicators.
11. Explore associations between mortality and demographic/socioeconomic variables.
12. Export analysis-ready data for SQL and BI tools.

## SQL workflow

The SQL script in `sql/covid19_analysis.sql` includes:

- coverage and data-quality checks
- cumulative cases and deaths by location
- cases/deaths per million
- case-fatality rate
- monthly case/death trends
- peak-date analysis
- testing and vaccination analysis
- public-health risk-factor extracts for visualization

A key modeling rule is that cumulative fields such as `total_cases` and `total_deaths` should **not be summed across dates**. Country snapshots should use the latest valid observation or an appropriate cumulative maximum after validating the source.

## Power BI dashboard plan

**Page 1 — Overview:** KPI cards, country filters, case trend and country comparisons.

**Page 2 — Cases & Mortality:** 7-day case/death trends, deaths per million, CFR and cases-versus-deaths comparisons.

**Page 3 — Testing, Vaccination & Healthcare:** positivity, testing, vaccination coverage, hospitalization and ICU trends.

**Page 4 — Public Health Risk Factors:** scatter plots for median age, GDP, hospital-bed capacity, diabetes prevalence and HDI versus population-adjusted mortality.

See `powerbi/README.md` for build steps and DAX examples.

## Tableau dashboard plan

The Tableau workflow mirrors the analytical questions while emphasizing interactive visual exploration:

- COVID-19 case trend
- deaths trend
- top represented countries
- deaths per million
- cases versus mortality
- vaccination trend
- demographic and healthcare risk-factor scatter plots

See `tableau/README.md` for worksheet and dashboard instructions.

## Data-quality limitation

The legacy workbook reviewed during development contained **65,535 data rows**, consistent with the old `.xls` worksheet row limit, and only a partial set of locations. Therefore, results from that workbook must be described as applying to the **locations represented in the file**, not as complete worldwide totals or rankings.

Before treating the project as a complete global analysis, replace the truncated legacy workbook with the complete authoritative source in CSV or `.xlsx` format and rerun the pipeline.

## Interpretation notes

- Population-adjusted rates are used alongside raw counts for fairer cross-country comparisons.
- Negative daily values may reflect reporting corrections and should be investigated rather than automatically deleted.
- Missing values should not automatically be converted to zero.
- Correlation and scatter plots show **association**, not causation.
- Cross-country comparisons can be affected by differences in testing, surveillance, reporting practices, demographics and timing.

## Portfolio value

This project demonstrates an end-to-end analytics workflow: **data quality → Python EDA → SQL analysis → BI visualization → public-health interpretation**. It is designed to show reproducible analytical reasoning rather than a collection of disconnected charts.
