# Chicago Crime Severity Prediction Analysis

## Dataset Overview
- **Total samples analyzed**: 96,428 hours
- **Features used**: 12 astrological variables
- **Training set**: 77,142 samples (80%)
- **Test set**: 19,286 samples (20%)

## Model Performance Comparison

| Model | R² Score | RMSE | MAE | Cross-Val RMSE |
|-------|----------|------|-----|----------------|
| **Random Forest** ⭐ | **0.1707** | **0.1926** | **0.1506** | **0.1930 (±0.0235)** |
| Gradient Boosting | 0.1140 | 0.1991 | 0.1565 | 0.1986 (±0.0182) |
| Linear Regression | 0.0398 | 0.2072 | 0.1631 | 0.2067 (±0.0230) |
| Ridge Regression | 0.0398 | 0.2072 | 0.1631 | 0.2067 (±0.0230) |

⭐ Best performing model

## Feature Importance Analysis

### Random Forest - Top 10 Features
1. **Ascendant Degree**: 31.23%
2. **Moon Phase**: 12.51%
3. **Uranus Longitude**: 8.38%
4. **Mercury Longitude**: 8.36%
5. **Sun Longitude**: 8.23%
6. **Jupiter Longitude**: 6.30%
7. **Venus Longitude**: 5.34%
8. **Pluto Longitude**: 5.03%
9. **Mars Longitude**: 4.95%
10. **Neptune Longitude**: 4.80%

### Gradient Boosting - Top 5 Features
1. **Ascendant Degree**: 34.57%
2. **Uranus Longitude**: 24.47%
3. **Sun Longitude**: 11.21%
4. **Jupiter Longitude**: 8.85%
5. **Saturn Longitude**: 6.92%

## Key Findings

### Best Model: Random Forest
- **Predictive Power**: R² of 0.1707 indicates the model explains ~17% of variance in crime severity
- **Error Metrics**: 
  - RMSE of 0.1926 shows relatively low prediction error
  - MAE of 0.1506 indicates average prediction is off by ~15% of severity scale
- **Stability**: Cross-validation RMSE of 0.1930 (±0.0235) shows consistent performance

### Most Influential Astrological Factors
1. **Ascendant Degree** dominates both Random Forest (31%) and Gradient Boosting (35%) models
2. **Moon Phase** shows significant influence (12.5%) in Random Forest
3. **Outer planets** (Uranus, Neptune, Pluto) collectively contribute substantial predictive power

### Model Comparison Insights
- **Tree-based models** (Random Forest, Gradient Boosting) significantly outperform linear models
- **Linear models** (Linear and Ridge Regression) show poor performance (R² < 0.04)
- Suggests non-linear relationships between astrological features and crime severity

## Astrological Features Analyzed
- **Solar**: Sun ecliptic longitude (0-360 degrees)
- **Lunar**: Moon phase cycles
- **Planetary**: All planets including Uranus, Neptune, and Pluto positions
- **Special**: Mercury retrograde status

## Conclusions
The analysis reveals measurable correlations between astrological positions and crime severity patterns in Chicago, with the ascendant degree and moon phase being the strongest predictors. While the predictive power is moderate (R² = 0.17), it suggests potential cyclical patterns worth further investigation.
