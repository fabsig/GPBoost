# Data sets used in demos

## gdp_european_regions.csv

The data is gross domestic product (GDP) data collected by Massimo Giannini, University of Rome Tor Vergata, from Eurostat. There is data for 242 European regions collected over two years 2000 and 2021. I.e., the number of data points is 484.

The response variable is

- (log) GDP / capita.

There are four predictor variables:

- L: log(share) of employment (empl/pop)
- K: log(fixed capita/population
- Pop: log(population)
- Edu: share of tertiary education

Furter, there are spatial coordinates:

- Long: region longitude
- Lat: region latitude

A spatial region ID:

- Group: the region ID (from 1 to 242)

And a spatial cluster ID:

- cl: identifies the cluster the region belongs
