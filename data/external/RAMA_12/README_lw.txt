
To TAO/TRITON, PIRATA, and RAMA data users:

Please acknowledge the GTMBA Project Office of NOAA/PMEL if you 
use these data in publications.  Also, we would appreciate receiving 
a preprint and/or reprint of publications utilizing the data for 
inclusion in the GTMBA Project bibliography.  Relevant publications
should be sent to:

Global Tropical Moored Buoy Array Project Office
NOAA/Pacific Marine Environmental Laboratory
7600 Sand Point Way NE
Seattle, WA 98115

Comments and questions should be directed to kenneth.connell@noaa.gov

Thank you.

Michael J. McPhaden
GTMBA Project Director
-------------------------------------------------------------------

The following topics will be covered in this readme file:

  1. Longwave Radiation
  2. Time Stamp
  3. 5-day, Monthly and Quarterly Averages
  4. Sampling, Sensors, and Moorings
  5. Quality Codes
  6. Source Codes
  7. References

1. Longwave Radiation:

Longwave radiation is available from several moorings 
in the Tropical Pacific since 2000, and at a few sites
in the Indian ocean, at daily and 2-minute resolution.

Daily average longwave radiation is computed as a 24 hour 
average, including day and night time values. Also included 
in 2 minute files is a time series of the standard deviation
of the sensor thermopile. Longwave radiation has units of 
Watts per square meter and is measured at a height of 3.5 
meters above mean sea level. (Instrument height is shown 
as a negative depth in data files).

2. Time Stamp:

Time associated with data that is processed and stored
on board by the moored buoy data acquisition system is
described for each sensor in the sampling protocol web page
at http://www.pmel.noaa.gov/gtmba/sampling.

Time associated with data that have been averaged in
post-processing represents the middle of the averaging
interval. For example, daily means computed from post
processed hourly or 10-min data are averaged over 24
hours starting at 0000 UTC and assigned an observation
time of 1200 UTC.

3. 5-day, Monthly and Quarterly Averages:

If you delivered 5-day, monthly, or quarterly averaged data
these definitions are relevant to your files:

5-Day: Average of data collected during consecutive five day 
intervals. A minimum of 2 daily values are required to compute 
a 5-day average.

Monthly: Average of all the data collected during each month.
A minimum of 15 daily values are required to compute a monthly 
average.

Quarterly: Average of 3 monthly values. A minimum of 2 monthly 
values are required to compute a quarterly average. 12 quarterly 
averages are computed for each year, one for each center month, 
which includes the previous month, the center month, and the 
next month in the average.

4. Sampling, Sensors, and Moorings:

For detailed information about sampling, sensors, and moorings,
see these web pages:

  http://www.pmel.noaa.gov/gtmba/sensor-specifications

  http://www.pmel.noaa.gov/gtmba/sampling

  http://www.pmel.noaa.gov/gtmba/moorings


5. Quality Codes:

In ascii format files organized by site, you will find data 
quality and source codes to the right of the data. In NetCDF 
format files organized by site, you will find quality and 
source variables with the same shape as the data.
These codes are defined below.

Using the quality codes you can tune your analysis to 
trade-off between quality and temporal/spatial coverage.
Quality code definitions are listed below

  0 = Datum Missing.

  1 = Highest Quality. Pre/post-deployment calibrations agree to within
  sensor specifications. In most cases, only pre-deployment calibrations 
  have been applied.

  2 = Default Quality. Default value for sensors presently deployed and 
  for sensors which were either not recovered, not calibratable when 
  recovered, or for which pre-deployment calibrations have been determined 
  to be invalid. In most cases, only pre-deployment calibrations have been 
  applied.

  3 = Adjusted Data. Pre/post calibrations differ, or original data do
  not agree with other data sources (e.g., other in situ data or 
  climatology), or original data are noisy. Data have been adjusted in 
  an attempt to reduce the error.

  4 = Lower Quality. Pre/post calibrations differ, or data do not agree
  with other data sources (e.g., other in situ data or climatology), or 
  data are noisy. Data could not be confidently adjusted to correct 
  for error.

  5 = Sensor or Tube Failed.

6. Source Codes:

  0 - No Sensor, No Data 
  1 - Real Time (Telemetered Mode)
  2 - Derived from Real Time
  3 - Temporally Interpolated from Real Time
  4 - Source Code Inactive at Present
  5 - Recovered from Instrument RAM (Delayed Mode)
  6 - Derived from RAM
  7 - Temporally Interpolated from RAM
  8 - Spatially Interpolated (e.g. vertically) from RAM

7. References:

For more information about TAO/TRITION, PIRATA, and RAMA, see

McPhaden, M.J., A.J. Busalacchi, R. Cheney, J.R. Donguy,K.S. 
Gage, D. Halpern, M. Ji, P. Julian, G. Meyers, G.T. Mitchum, 
P.P. Niiler, J. Picaut, R.W. Reynolds, N. Smith, K. Takeuchi, 
1998: The Tropical Ocean-Global Atmosphere (TOGA) observing 
system:  A decade of progress. J. Geophys. Res., 103, 14,
169-14,240.

Bourles, B., R. Lumpkin, M.J. McPhaden, F. Hernandez, P. Nobre, 
E.Campos, L. Yu, S. Planton, A. Busalacchi, A.D. Moura, J. 
Servain, and J. Trotte, 2008: The PIRATA Program: History, 
Accomplishments, and Future Directions. Bull. Amer. Meteor. 
Soc., 89, 1111-1125.

McPhaden, M.J., G. Meyers, K. Ando, Y. Masumoto, V.S.N. Murty, M.
Ravichandran, F. Syamsudin, J. Vialard, L. Yu, and W. Yu, 2009: RAMA: The
Research Moored Array for African-Asian-Australian Monsoon Analysis and
Prediction. Bull. Am. Meteorol. Soc., 90, 459-480,
doi:10.1175/2008BAMS2608.1
