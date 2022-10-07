---
layout: dashboard
---

[Click here to see the full size Dashboard](https://public.tableau.com/app/profile/paul.fisher5714/viz/CovidDashboard_16650690608120/Dashboard1#1)

# Tableau creation

Here is an example of a project I did looking at COVID statistics using Tableau. Data are available here:

[Link to data](https://ourworldindata.org/covid-deaths)

The general idea was to get a display that would give us up-to-date (as of 05/10/2022) visualised statistics of global
COVID death-rate and infections. The **Perfect Population Infected** figure also contains projects for the future months.

# Data aquisition in SQL

The data were downloaded and stored in excel files and cleaned using Microsoft SQL Server Management Studio (SSMS)

Here are some examples of the queries used to explore the dataset:

```sql
with populationvsvaccination (Continent, Location, Date, Population, New_Vaccinations, RollingPeopleVaccinated)
as
(
select death.continent, death.location, death.date, death.population, vaccine.new_vaccinations,
    sum(cast(vaccine.new_vaccinations as int)) 
    over (partition by death.Location 
    order by death.location, death.Date) as RollingPeopleVaccinated
from [Portfolio Project]..CovidDeaths death
join [Portfolio Project]..CovidVaccinations vaccine
	on dea.location = vaccine.location
	and dea.date = vaccine.date
where dea.continent is not null -- this is simply because of an irregularity in the dataset between location/continent
)
select *, (RollingPeopleVaccinated/Population)*100 as RollingPeopleVaccinatedPercentage
from populationvsvaccination
```

Here is an example of a common-table-expression (CTE) that was used to perform a Partition By query.
The data had been separated into two excel files (deaths and vaccinations) for simplicity. The Partition By creates a 
rolling total of vaccinations iteratively ordered by date at each location.

The CTE is in order to generate the **population** vs **vaccination status** rolling percentage within one query.