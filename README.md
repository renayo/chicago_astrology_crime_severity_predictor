# Chicago Crime Severity Prediction Analysis

## Dataset Overview
- **Total samples analyzed**: 96,428 hours
- **Features used**: 9 astrological variables
- **Training set**: 77,142 samples (80%)
- **Test set**: 19,286 samples (20%)

## Model Performance Comparison

| Model | R² Score | RMSE | MAE | Cross-Val RMSE |
|-------|----------|------|-----|----------------|
| **Random Forest** ⭐ | **0.1748** | **0.1921** | **0.1502** | **0.1928 (±0.0240)** |
| Gradient Boosting | 0.1246 | 0.1979 | 0.1555 | 0.1976 (±0.0199) |
| Linear Regression | 0.0281 | 0.2085 | 0.1642 | 0.2077 (±0.0227) |
| Ridge Regression | 0.0281 | 0.2085 | 0.1642 | 0.2077 (±0.0227) |

⭐ Best performing model

## Feature Importance Analysis

### Random Forest - All 9 Features Ranked
1. **Ascendant Degree**: 31.64%
2. **Moon Phase**: 13.57%
3. **Sun Longitude**: 11.19%
4. **Saturn Longitude**: 10.00%
5. **Mercury Longitude**: 9.97%
6. **Jupiter Longitude**: 8.24%
7. **Mars Longitude**: 7.79%
8. **Venus Longitude**: 7.31%
9. **Mercury Retrograde**: 0.28%

### Gradient Boosting - Top 5 Features
1. **Ascendant Degree**: 41.99%
2. **Saturn Longitude**: 24.76%
3. **Sun Longitude**: 12.14%
4. **Jupiter Longitude**: 9.52%
5. **Mars Longitude**: 6.42%

## Key Findings

### Best Model: Random Forest
- **Predictive Power**: R² of 0.1748 indicates the model explains ~17.5% of variance in crime severity
- **Error Metrics**: 
  - RMSE of 0.1921 shows relatively low prediction error
  - MAE of 0.1502 indicates average prediction is off by ~15% of severity scale
- **Stability**: Cross-validation RMSE of 0.1928 (±0.0240) demonstrates consistent performance

### Most Influential Astrological Factors
1. **Ascendant Degree** strongly dominates both models (31.6% RF, 42% GB)
2. **Moon Phase** shows substantial influence (13.6%) in Random Forest
3. **Classical Planets** (Sun, Saturn, Mercury) collectively contribute significant predictive power
4. **Mercury Retrograde** has minimal impact (0.28%)

### Model Comparison Insights
- **Tree-based models** vastly outperform linear approaches (6x better R²)
- **Linear models** show very poor performance (R² < 0.03)
- **Non-linear relationships** clearly exist between astrological features and crime patterns
- Random Forest slightly outperforms Gradient Boosting despite using fewer features

## Performance vs. 12-Feature Model
Comparing this 9-feature model to a previous 12-feature analysis:
- **R² Score**: 0.1748 (9 features) vs 0.1707 (12 features)
- **RMSE**: 0.1921 (9 features) vs 0.1926 (12 features)
- **Conclusion**: Reducing features from 12 to 9 actually improved performance slightly

## Astrological Features Analyzed
- **Solar**: Sun ecliptic longitude (0-360 degrees)
- **Lunar**: Moon phase cycles
- **Planetary**: All traditional planets plus Saturn's prominent role
- **Special**: Mercury retrograde status (minimal predictive value)

## Conclusions
The analysis demonstrates that a simplified 9-feature model achieves better predictive performance than a more complex 12-feature version. The ascendant degree remains the dominant predictor (31.6%), followed by moon phase (13.6%) and sun position (11.2%). Saturn shows unexpectedly high importance in the Gradient Boosting model (24.8%), suggesting potential cyclical patterns related to this planet's position. The minimal impact of Mercury retrograde (0.28%) challenges popular astrological beliefs about its influence on earthly events.
