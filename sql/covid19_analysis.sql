/*
COVID-19 Public Health Analytics
SQL Server analysis workflow

Expected table: dbo.CovidData
Important: filter regional/world aggregates when country-level comparisons are required.
*/

-- 1. Dataset coverage
SELECT
    COUNT(*) AS total_rows,
    COUNT(DISTINCT location) AS locations,
    MIN([date]) AS start_date,
    MAX([date]) AS end_date
FROM dbo.CovidData;

-- 2. Duplicate location-date observations
SELECT location, [date], COUNT(*) AS row_count
FROM dbo.CovidData
GROUP BY location, [date]
HAVING COUNT(*) > 1
ORDER BY row_count DESC;

-- 3. Negative daily values / reporting corrections
SELECT location, [date], new_cases, new_deaths
FROM dbo.CovidData
WHERE new_cases < 0 OR new_deaths < 0
ORDER BY [date], location;

-- 4. Latest observation per location
WITH Latest AS (
    SELECT *,
           ROW_NUMBER() OVER (
               PARTITION BY location
               ORDER BY [date] DESC
           ) AS rn
    FROM dbo.CovidData
)
SELECT location, [date], total_cases, total_deaths, population
FROM Latest
WHERE rn = 1
ORDER BY total_cases DESC;

-- 5. Highest cumulative cases
WITH Latest AS (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY location ORDER BY [date] DESC) AS rn
    FROM dbo.CovidData
)
SELECT TOP (10)
    location,
    total_cases
FROM Latest
WHERE rn = 1 AND total_cases IS NOT NULL
ORDER BY total_cases DESC;

-- 6. Highest cumulative deaths
WITH Latest AS (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY location ORDER BY [date] DESC) AS rn
    FROM dbo.CovidData
)
SELECT TOP (10)
    location,
    total_deaths
FROM Latest
WHERE rn = 1 AND total_deaths IS NOT NULL
ORDER BY total_deaths DESC;

-- 7. Population-adjusted burden
WITH Latest AS (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY location ORDER BY [date] DESC) AS rn
    FROM dbo.CovidData
)
SELECT
    location,
    total_cases_per_million,
    total_deaths_per_million
FROM Latest
WHERE rn = 1
ORDER BY total_deaths_per_million DESC;

-- 8. Validate case-fatality rate
WITH Latest AS (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY location ORDER BY [date] DESC) AS rn
    FROM dbo.CovidData
)
SELECT
    location,
    total_cases,
    total_deaths,
    100.0 * total_deaths / NULLIF(total_cases, 0) AS calculated_cfr
FROM Latest
WHERE rn = 1
ORDER BY calculated_cfr DESC;

-- 9. Monthly cases and deaths
SELECT
    location,
    YEAR([date]) AS [year],
    MONTH([date]) AS [month],
    SUM(new_cases) AS monthly_cases,
    SUM(new_deaths) AS monthly_deaths
FROM dbo.CovidData
GROUP BY location, YEAR([date]), MONTH([date])
ORDER BY location, [year], [month];

-- 10. Peak case days
WITH RankedCases AS (
    SELECT
        location,
        [date],
        new_cases,
        ROW_NUMBER() OVER (
            PARTITION BY location
            ORDER BY new_cases DESC, [date]
        ) AS rn
    FROM dbo.CovidData
    WHERE new_cases IS NOT NULL
)
SELECT location, [date], new_cases
FROM RankedCases
WHERE rn = 1
ORDER BY new_cases DESC;

-- 11. Peak death days
WITH RankedDeaths AS (
    SELECT
        location,
        [date],
        new_deaths,
        ROW_NUMBER() OVER (
            PARTITION BY location
            ORDER BY new_deaths DESC, [date]
        ) AS rn
    FROM dbo.CovidData
    WHERE new_deaths IS NOT NULL
)
SELECT location, [date], new_deaths
FROM RankedDeaths
WHERE rn = 1
ORDER BY new_deaths DESC;

-- 12. Testing / positivity trend
SELECT
    location,
    [date],
    new_tests,
    positive_rate,
    tests_per_case,
    new_cases
FROM dbo.CovidData
WHERE new_tests IS NOT NULL OR positive_rate IS NOT NULL
ORDER BY location, [date];

-- 13. Vaccination trend
SELECT
    location,
    [date],
    people_vaccinated,
    people_fully_vaccinated,
    total_boosters,
    vaccination_rate
FROM dbo.CovidData
WHERE people_vaccinated IS NOT NULL
   OR people_fully_vaccinated IS NOT NULL
ORDER BY location, [date];

-- 14. Hospital and ICU burden
SELECT
    location,
    [date],
    hosp_patients,
    icu_patients,
    weekly_hosp_admissions,
    weekly_icu_admissions
FROM dbo.CovidData
WHERE hosp_patients IS NOT NULL
   OR icu_patients IS NOT NULL
ORDER BY location, [date];

-- 15. Risk-factor dataset for BI/statistical exploration
WITH Latest AS (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY location ORDER BY [date] DESC) AS rn
    FROM dbo.CovidData
)
SELECT
    location,
    total_deaths_per_million,
    median_age,
    aged_65_older,
    gdp_per_capita,
    diabetes_prevalence,
    hospital_beds_per_thousand,
    life_expectancy,
    human_development_index
FROM Latest
WHERE rn = 1
ORDER BY location;
