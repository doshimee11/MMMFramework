"""
Data Preprocessing for MMM
- Feature engineering
- Train/test split
- Scaling and normalization
- Data validation
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODELING_CONFIG, MARKETING_CHANNELS


def load_data(filepath='data/mmm_data.csv'):
    """Load MMM data"""
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    return df


def create_time_features(df):
    """Create time-based features"""
    df = df.copy()
    
    # Extract date components
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['week'] = df['date'].dt.isocalendar().week
    df['day_of_year'] = df['date'].dt.dayofyear
    
    # Cyclical encoding for seasonality
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['week_sin'] = np.sin(2 * np.pi * df['week'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week'] / 52)
    
    # Time index (for trend)
    df['time_index'] = np.arange(len(df))
    
    return df


def create_lag_features(df, channels, lags=[1, 2, 4]):
    """Create lagged features for spend"""
    df = df.copy()
    
    for channel in channels:
        spend_col = f'{channel}_spend'
        for lag in lags:
            df[f'{spend_col}_lag{lag}'] = df[spend_col].shift(lag)
    
    # Fill NaN values with 0 for lag features
    lag_cols = [col for col in df.columns if '_lag' in col]
    df[lag_cols] = df[lag_cols].fillna(0)
    
    return df


def create_rolling_features(df, channels, windows=[4, 8, 12]):
    """Create rolling average features"""
    df = df.copy()
    
    for channel in channels:
        spend_col = f'{channel}_spend'
        for window in windows:
            df[f'{spend_col}_ma{window}'] = df[spend_col].rolling(window=window, min_periods=1).mean()
    
    return df


def prepare_features_target(df, target='total_sales'):
    """
    Prepare feature matrix and target vector
    
    Features include:
    - Marketing spend (all channels)
    - Time features (seasonality, trend)
    - External factors (holidays, promotions, competitors)
    """
    
    # Marketing spend features
    spend_features = [f'{channel}_spend' for channel in MARKETING_CHANNELS.keys()]
    
    # Time features
    time_features = ['time_index', 'month_sin', 'month_cos', 'week_sin', 'week_cos']
    
    # External factors
    external_features = ['is_holiday', 'competitor_active', 'has_promotion']
    
    # Combine all features
    feature_columns = spend_features + time_features + external_features
    
    # Check if all features exist
    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
        feature_columns = [col for col in feature_columns if col in df.columns]
    
    X = df[feature_columns].copy()
    y = df[target].copy()
    
    return X, y, feature_columns


def split_data(df, method='time_series', test_size=0.2):
    """
    Split data into train and test sets
    
    Methods:
    - time_series: Use most recent data for test (respects time order)
    - random: Random split (for cross-validation)
    """
    
    if method == 'time_series':
        # Time series split - use most recent data for test
        split_idx = int(len(df) * (1 - test_size))
        
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        print(f"Time series split:")
        print(f"  Train: {train_df['date'].min()} to {train_df['date'].max()} ({len(train_df)} periods)")
        print(f"  Test: {test_df['date'].min()} to {test_df['date'].max()} ({len(test_df)} periods)")
    
    else:
        # Random split
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=MODELING_CONFIG['random_state']
        )
        
        print(f"Random split:")
        print(f"  Train: {len(train_df)} periods")
        print(f"  Test: {len(test_df)} periods")
    
    return train_df, test_df


def scale_features(X_train, X_test, method='standard'):
    """
    Scale features for modeling
    
    Methods:
    - standard: Standardize to mean=0, std=1
    - minmax: Scale to [0, 1] range
    - none: No scaling
    """
    
    if method == 'standard':
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert back to DataFrame
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    elif method == 'minmax':
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    else:
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        scaler = None
    
    return X_train_scaled, X_test_scaled, scaler


def validate_data(df):
    """Run data validation checks"""
    
    print("\n" + "="*70)
    print("DATA VALIDATION")
    print("="*70)
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("\n⚠️  Missing values found:")
        print(missing[missing > 0])
    else:
        print("\n✓ No missing values")
    
    # Check for negative spend
    spend_cols = [f'{channel}_spend' for channel in MARKETING_CHANNELS.keys()]
    negative_spend = (df[spend_cols] < 0).any()
    if negative_spend.any():
        print("\n⚠️  Negative spend values found:")
        print(negative_spend[negative_spend])
    else:
        print("✓ No negative spend values")
    
    # Check for negative sales
    if (df['total_sales'] < 0).any():
        print("\n⚠️  Negative sales values found!")
        print(f"  Count: {(df['total_sales'] < 0).sum()}")
    else:
        print("✓ No negative sales values")
    
    # Check data range
    print(f"\nData range:")
    print(f"  Dates: {df['date'].min()} to {df['date'].max()}")
    print(f"  Periods: {len(df)}")
    
    # Check for duplicates
    duplicates = df.duplicated(subset=['date']).sum()
    if duplicates > 0:
        print(f"\n⚠️  {duplicates} duplicate dates found!")
    else:
        print("✓ No duplicate dates")
    
    print("\n" + "="*70)


def preprocess_pipeline(filepath='data/mmm_data.csv'):
    """Complete preprocessing pipeline"""
    
    print("\n" + "="*70)
    print("PREPROCESSING PIPELINE")
    print("="*70)
    
    # Load data
    print("\n1. Loading data...")
    df = load_data(filepath)
    print(f"   Loaded {len(df)} periods")
    
    # Validate data
    print("\n2. Validating data...")
    validate_data(df)
    
    # Create features
    print("\n3. Creating time features...")
    df = create_time_features(df)
    
    # Prepare features and target
    print("\n4. Preparing features and target...")
    X, y, feature_columns = prepare_features_target(df)
    print(f"   Features: {len(feature_columns)}")
    print(f"   Target: total_sales")
    
    # Split data
    print("\n5. Splitting data...")
    train_df, test_df = split_data(df, method=MODELING_CONFIG['validation_method'])
    
    # Prepare train/test sets
    X_train, y_train, _ = prepare_features_target(train_df)
    X_test, y_test, _ = prepare_features_target(test_df)
    
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE")
    print("="*70)
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Features: {X_train.shape[1]}")
    
    return {
        'df': df,
        'train_df': train_df,
        'test_df': test_df,
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'feature_columns': feature_columns,
    }


if __name__ == "__main__":
    results = preprocess_pipeline()
    
    print("\n\nFeature columns:")
    for i, col in enumerate(results['feature_columns'], 1):
        print(f"{i}. {col}")
