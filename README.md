# Chicago astrology crime severity predictor

================================================================================
CHICAGO CRIME SEVERITY PREDICTOR
Astrological Feature Analysis (2014-2024)
================================================================================

============================================================
DATA PREPARATION
============================================================
============================================================
FETCHING CHICAGO CRIME DATA (2014-2024)
============================================================

Fetching batch 1 (offset: 0)...
  Retrieved 50,000 records

Fetching batch 2 (offset: 50000)...
  Retrieved 50,000 records

Fetching batch 3 (offset: 100000)...
  Error: HTTP 500

============================================================
TOTAL RECORDS FETCHED: 100,000
============================================================

Processing crime data...
✓ Processed 100,000 valid records
  Date range: 2014-01-01 00:00:00 to 2014-05-21 19:00:00

Mapping crime severity...
Calculating hourly averages...
✓ Created 3,380 hourly data points

============================================================
CALCULATING ASTROLOGICAL FEATURES
============================================================
Processing 3,380 hours...
This may take several minutes...

  Progress: 0/3,380 hours (0.0%)
  Progress: 1,000/3,380 hours (29.6%)
  Progress: 2,000/3,380 hours (59.2%)
  Progress: 3,000/3,380 hours (88.8%)

✓ Calculated features for all 3,380 hours

============================================================
MODEL TRAINING
============================================================
Features: 13
Samples: 3,380
Training set: 2,704 samples
Test set: 676 samples

Training models:

  Linear Regression...
    R² Score: 0.0508
    RMSE: 0.2442

  Ridge Regression...
    R² Score: 0.0511
    RMSE: 0.2441

  Random Forest...
    R² Score: 0.2046
    RMSE: 0.2235

  Gradient Boosting...
    R² Score: 0.1861
    RMSE: 0.2261

================================================================================
CHICAGO CRIME SEVERITY PREDICTION RESULTS
================================================================================

Linear Regression:
  Mean Absolute Error: 0.1881
  Mean Squared Error: 0.0596
  Root Mean Squared Error: 0.2442
  R² Score: 0.0508
  Cross-Val RMSE: 0.2528 (±0.0724)

Ridge Regression:
  Mean Absolute Error: 0.1878
  Mean Squared Error: 0.0596
  Root Mean Squared Error: 0.2441
  R² Score: 0.0511
  Cross-Val RMSE: 0.2525 (±0.0727)

Random Forest:
  Mean Absolute Error: 0.1724
  Mean Squared Error: 0.0500
  Root Mean Squared Error: 0.2235
  R² Score: 0.2046
  Cross-Val RMSE: 0.2408 (±0.0585)

  Top 5 Important Features:
    - ascendant_degree: 0.3771
    - moon_phase: 0.1138
    - venus_longitude: 0.0677
    - mercury_longitude: 0.0624
    - jupiter_longitude: 0.0614

Gradient Boosting:
  Mean Absolute Error: 0.1707
  Mean Squared Error: 0.0511
  Root Mean Squared Error: 0.2261
  R² Score: 0.1861
  Cross-Val RMSE: 0.2386 (±0.0668)

  Top 5 Important Features:
    - ascendant_degree: 0.5301
    - pluto_longitude: 0.0844
    - uranus_longitude: 0.0646
    - moon_phase: 0.0528
    - sun_longitude: 0.0490

================================================================================
BEST PERFORMING MODEL
================================================================================

Model: Random Forest
R² Score: 0.2046
RMSE: 0.2235
MAE: 0.1724

Top 10 Important Features:
  ascendant_degree    :  37.71%
  moon_phase          :  11.38%
  venus_longitude     :   6.77%
  mercury_longitude   :   6.24%
  jupiter_longitude   :   6.14%
  sun_longitude       :   5.92%
  pluto_longitude     :   5.54%
  mars_longitude      :   5.32%
  saturn_longitude    :   5.02%
  neptune_longitude   :   4.83%

================================================================================
ANALYSIS COMPLETE
================================================================================

Astrological features analyzed:
  • Temporal: Week of year
  • Solar: Sun ecliptic longitude (0-360 degrees)
  • Lunar: Moon phase
  • Planetary: All planets including Uranus, Neptune, and Pluto
  • Special: Mercury retrograde status, ascendant degree

Models trained:
  • Linear Regression
  • Ridge Regression
  • Random Forest
  • Gradient Boosting
