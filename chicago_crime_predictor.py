import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import ephem
import warnings
warnings.filterwarnings('ignore')

# Severity mapping from the provided file
severity_map = {
    'HOMICIDE': 5,
    'CRIMINAL SEXUAL ASSAULT': 5,
    'SEX OFFENSE': 5,
    'ROBBERY': 4,
    'BATTERY': 4,
    'ASSAULT': 4,
    'BURGLARY': 3,
    'MOTOR VEHICLE THEFT': 3,
    'THEFT': 2,
    'CRIMINAL DAMAGE': 2,
    'DECEPTIVE PRACTICE': 1,
    'OTHER OFFENSE': 2.5
}

class CrimeDataFetcher:
    """Fetch crime data from Chicago open data portal"""
    
    def __init__(self):
        self.chicago_info = {
            'url': 'https://data.cityofchicago.org/resource/ijzp-q8t2.json',
            'lat': 41.8781,
            'lon': -87.6298
        }
    
    def fetch_chicago_data(self):
        """Fetch Chicago crime data in batches"""
        print("="*60)
        print("FETCHING CHICAGO CRIME DATA (2014-2024)")
        print("="*60)
        
        all_data = []
        offset = 0
        batch_size = 50000
        batch_num = 1
        
        while True:
            params = {
                '$limit': batch_size,
                '$offset': offset,
                '$where': "date >= '2014-01-01T00:00:00' AND date <= '2024-12-31T23:59:59'",
                '$select': 'date,primary_type'
            }
            
            try:
                print(f"\nFetching batch {batch_num} (offset: {offset})...")
                response = requests.get(self.chicago_info['url'], params=params)
                
                if response.status_code != 200:
                    print(f"  Error: HTTP {response.status_code}")
                    break
                
                data = response.json()
                
                if not data:
                    print("  No more data to fetch")
                    break
                
                print(f"  Retrieved {len(data):,} records")
                all_data.extend(data)
                
                if len(data) < batch_size:
                    print("\n✓ Reached end of dataset")
                    break
                
                offset += batch_size
                batch_num += 1
                
            except Exception as e:
                print(f"\n✗ Error fetching data: {e}")
                break
        
        print(f"\n{'='*60}")
        print(f"TOTAL RECORDS FETCHED: {len(all_data):,}")
        print(f"{'='*60}\n")
        
        return all_data
    
    def process_chicago_data(self, data):
        """Process Chicago crime data"""
        print("Processing crime data...")
        
        df = pd.DataFrame(data)
        if df.empty:
            print("✗ No data to process")
            return pd.DataFrame()
        
        # Parse datetime
        df['datetime'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['datetime'])
        
        # Clean crime types
        df['crime_type'] = df['primary_type'].str.upper().str.strip()
        
        print(f"✓ Processed {len(df):,} valid records")
        print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        
        return df[['datetime', 'crime_type']]

class AstrologicalCalculator:
    """Calculate astrological features for given datetime and location"""
    
    def __init__(self, lat, lon):
        self.observer = ephem.Observer()
        self.observer.lat = str(lat)
        self.observer.lon = str(lon)
        
    def calculate_features(self, dt):
        """Calculate all astrological features for a given datetime"""
        self.observer.date = dt
        
        features = {}
        
        # Basic temporal features
        features['week_of_year'] = dt.isocalendar()[1]
        
        # Moon phase (0 = new moon, 0.5 = full moon)
        moon = ephem.Moon(self.observer)
        features['moon_phase'] = moon.moon_phase
        
        # Solar features - ONLY sun longitude
        sun = ephem.Sun(self.observer)
        features['sun_longitude'] = float(ephem.Ecliptic(sun).lon) * 180 / np.pi
        
        # Mercury retrograde
        features['mercury_retrograde'] = self.is_mercury_retrograde(dt)
        
        # All planetary positions (including outer planets)
        planets = {
            'mercury': ephem.Mercury(self.observer),
            'venus': ephem.Venus(self.observer),
            'mars': ephem.Mars(self.observer),
            'jupiter': ephem.Jupiter(self.observer),
            'saturn': ephem.Saturn(self.observer),
            'uranus': ephem.Uranus(self.observer),
            'neptune': ephem.Neptune(self.observer),
            'pluto': ephem.Pluto(self.observer)
        }
        
        for name, planet in planets.items():
            features[f'{name}_longitude'] = float(ephem.Ecliptic(planet).lon) * 180 / np.pi
        
        # Approximate ascendant degree
        lst = self.observer.sidereal_time()
        features['ascendant_degree'] = float(lst) * 15  # Convert hours to degrees
        
        return features
    
    def is_mercury_retrograde(self, dt):
        """Check if Mercury is in retrograde"""
        retrograde_periods = [
            (datetime(2014, 2, 6), datetime(2014, 2, 28)),
            (datetime(2014, 6, 7), datetime(2014, 7, 2)),
            (datetime(2014, 10, 4), datetime(2014, 10, 25)),
            (datetime(2015, 1, 21), datetime(2015, 2, 11)),
            (datetime(2015, 5, 19), datetime(2015, 6, 11)),
            (datetime(2015, 9, 17), datetime(2015, 10, 9)),
            (datetime(2016, 1, 5), datetime(2016, 1, 25)),
            (datetime(2016, 4, 28), datetime(2016, 5, 22)),
            (datetime(2016, 8, 30), datetime(2016, 9, 22)),
            (datetime(2016, 12, 19), datetime(2017, 1, 8)),
            (datetime(2017, 4, 10), datetime(2017, 5, 3)),
            (datetime(2017, 8, 13), datetime(2017, 9, 5)),
            (datetime(2017, 12, 3), datetime(2017, 12, 23)),
            (datetime(2018, 3, 23), datetime(2018, 4, 15)),
            (datetime(2018, 7, 26), datetime(2018, 8, 19)),
            (datetime(2018, 11, 17), datetime(2018, 12, 7)),
            (datetime(2019, 3, 5), datetime(2019, 3, 28)),
            (datetime(2019, 7, 8), datetime(2019, 8, 1)),
            (datetime(2019, 10, 31), datetime(2019, 11, 20)),
            (datetime(2020, 2, 17), datetime(2020, 3, 10)),
            (datetime(2020, 6, 18), datetime(2020, 7, 12)),
            (datetime(2020, 10, 14), datetime(2020, 11, 3)),
            (datetime(2021, 1, 30), datetime(2021, 2, 21)),
            (datetime(2021, 5, 30), datetime(2021, 6, 23)),
            (datetime(2021, 9, 27), datetime(2021, 10, 18)),
            (datetime(2022, 1, 14), datetime(2022, 2, 4)),
            (datetime(2022, 5, 10), datetime(2022, 6, 3)),
            (datetime(2022, 9, 10), datetime(2022, 10, 2)),
            (datetime(2022, 12, 29), datetime(2023, 1, 18)),
            (datetime(2023, 4, 21), datetime(2023, 5, 15)),
            (datetime(2023, 8, 23), datetime(2023, 9, 15)),
            (datetime(2023, 12, 13), datetime(2024, 1, 2)),
            (datetime(2024, 4, 2), datetime(2024, 4, 25)),
            (datetime(2024, 8, 5), datetime(2024, 8, 28)),
            (datetime(2024, 11, 26), datetime(2024, 12, 15))
        ]
        
        for start, end in retrograde_periods:
            if start <= dt <= end:
                return 1
        return 0

class ChicagoCrimeSeverityPredictor:
    """Main class for Chicago crime severity prediction"""
    
    def __init__(self):
        self.fetcher = CrimeDataFetcher()
        self.models = {}
        self.results = {}
        
    def prepare_data(self):
        """Prepare Chicago crime data with astrological features"""
        print("\n" + "="*60)
        print("DATA PREPARATION")
        print("="*60)
        
        # Fetch data
        data = self.fetcher.fetch_chicago_data()
        
        if not data:
            print("✗ No data fetched")
            return None
        
        # Process data
        df = self.fetcher.process_chicago_data(data)
        
        if df.empty:
            print("✗ No valid data after processing")
            return None
        
        # Map crime types to severity
        print("\nMapping crime severity...")
        df['severity'] = df['crime_type'].apply(
            lambda x: max([severity_map.get(key, 2.5) 
                          for key in severity_map.keys() 
                          if key in str(x)] or [2.5])
        )
        
        # Calculate hourly averages
        print("Calculating hourly averages...")
        df['hour'] = df['datetime'].dt.floor('H')
        hourly_severity = df.groupby('hour')['severity'].mean().reset_index()
        
        print(f"✓ Created {len(hourly_severity):,} hourly data points")
        
        # Calculate astrological features
        print("\n" + "="*60)
        print("CALCULATING ASTROLOGICAL FEATURES")
        print("="*60)
        print(f"Processing {len(hourly_severity):,} hours...")
        print("This may take several minutes...\n")
        
        calc = AstrologicalCalculator(
            self.fetcher.chicago_info['lat'], 
            self.fetcher.chicago_info['lon']
        )
        
        features_list = []
        total_hours = len(hourly_severity)
        
        for idx, row in hourly_severity.iterrows():
            if idx % 1000 == 0:
                progress = (idx / total_hours) * 100
                print(f"  Progress: {idx:,}/{total_hours:,} hours ({progress:.1f}%)")
            
            features = calc.calculate_features(row['hour'])
            features['severity'] = row['severity']
            features_list.append(features)
        
        print(f"\n✓ Calculated features for all {total_hours:,} hours")
        
        return pd.DataFrame(features_list)
    
    def train_models(self, df):
        """Train multiple models for comparison"""
        print("\n" + "="*60)
        print("MODEL TRAINING")
        print("="*60)
        
        # Prepare features and target
        X = df.drop('severity', axis=1)
        y = df['severity']
        
        print(f"Features: {X.shape[1]}")
        print(f"Samples: {X.shape[0]:,}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Training set: {len(X_train):,} samples")
        print(f"Test set: {len(X_test):,} samples")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        print("\nTraining models:")
        for name, model in models.items():
            print(f"\n  {name}...")
            
            # Use scaled data for linear models
            if 'Linear' in name or 'Ridge' in name:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                # Cross-validation
                cv_scores = cross_val_score(
                    model, X_train_scaled, y_train, 
                    cv=5, scoring='neg_mean_squared_error'
                )
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Cross-validation
                cv_scores = cross_val_score(
                    model, X_train, y_train, 
                    cv=5, scoring='neg_mean_squared_error'
                )
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'cv_rmse': np.sqrt(-cv_scores.mean()),
                'cv_std': np.sqrt(cv_scores.std())
            }
            
            print(f"    R² Score: {r2:.4f}")
            print(f"    RMSE: {rmse:.4f}")
            
            # Feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                results[name]['feature_importance'] = importance
        
        self.models = models
        self.results = results
        
        return results
    
    def print_statistics(self):
        """Print comprehensive statistics"""
        print("\n" + "="*80)
        print("CHICAGO CRIME SEVERITY PREDICTION RESULTS")
        print("="*80)
        
        for model_name, metrics in self.results.items():
            print(f"\n{model_name}:")
            print(f"  Mean Absolute Error: {metrics['mae']:.4f}")
            print(f"  Mean Squared Error: {metrics['mse']:.4f}")
            print(f"  Root Mean Squared Error: {metrics['rmse']:.4f}")
            print(f"  R² Score: {metrics['r2']:.4f}")
            print(f"  Cross-Val RMSE: {metrics['cv_rmse']:.4f} (±{metrics['cv_std']:.4f})")
            
            if 'feature_importance' in metrics:
                print(f"\n  Top 5 Important Features:")
                for idx, row in metrics['feature_importance'].head().iterrows():
                    print(f"    - {row['feature']}: {row['importance']:.4f}")
        
        # Best model summary
        print("\n" + "="*80)
        print("BEST PERFORMING MODEL")
        print("="*80)
        
        best_model = max(self.results.items(), key=lambda x: x[1]['r2'])
        print(f"\nModel: {best_model[0]}")
        print(f"R² Score: {best_model[1]['r2']:.4f}")
        print(f"RMSE: {best_model[1]['rmse']:.4f}")
        print(f"MAE: {best_model[1]['mae']:.4f}")
        
        if 'feature_importance' in best_model[1]:
            print(f"\nTop 10 Important Features:")
            for idx, row in best_model[1]['feature_importance'].head(10).iterrows():
                importance_pct = row['importance'] * 100
                print(f"  {row['feature']:20s}: {importance_pct:6.2f}%")

def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("CHICAGO CRIME SEVERITY PREDICTOR")
    print("Astrological Feature Analysis (2014-2024)")
    print("="*80)
    
    predictor = ChicagoCrimeSeverityPredictor()
    
    # Prepare data
    df = predictor.prepare_data()
    
    if df is not None and not df.empty:
        # Train models
        predictor.train_models(df)
        
        # Print statistics
        predictor.print_statistics()
    else:
        print("\n✗ Unable to complete analysis due to data issues")
        return None
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nAstrological features analyzed:")
    print("  • Temporal: Week of year")
    print("  • Solar: Sun ecliptic longitude (0-360 degrees)")
    print("  • Lunar: Moon phase")
    print("  • Planetary: All planets including Uranus, Neptune, and Pluto")
    print("  • Special: Mercury retrograde status, ascendant degree")
    print("\nModels trained:")
    print("  • Linear Regression")
    print("  • Ridge Regression")
    print("  • Random Forest")
    print("  • Gradient Boosting")
    
    return predictor

if __name__ == "__main__":
    predictor = main()
