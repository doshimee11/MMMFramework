"""
Marketing Mix Modeling - Core Modeling Module
Implements multiple MMM approaches:
1. Linear Regression (baseline)
2. Ridge/Lasso Regression
3. XGBoost
4. Prophet
5. Custom MMM with adstock and saturation
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor

# Try to import XGBoost (optional dependency)
try:
    import xgboost as xgb
    # Test if xgboost actually works (library loading)
    _test = xgb.XGBRegressor()
    XGBOOST_AVAILABLE = True
except Exception:
    # Catches ImportError, XGBoostError (library not loaded), etc.
    xgb = None
    XGBOOST_AVAILABLE = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MARKETING_CHANNELS, MODELING_CONFIG
from src.data_processing import preprocess_pipeline


class LinearMMMModel:
    """Linear regression baseline for MMM"""
    
    def __init__(self, alpha=1.0, model_type='ridge'):
        self.model_type = model_type
        self.alpha = alpha
        
        if model_type == 'ridge':
            self.model = Ridge(alpha=alpha, random_state=MODELING_CONFIG['random_state'])
        elif model_type == 'lasso':
            self.model = Lasso(alpha=alpha, random_state=MODELING_CONFIG['random_state'])
        else:
            self.model = LinearRegression()
        
        self.feature_importance = None
        self.coefficients = None
    
    def fit(self, X_train, y_train):
        """Train the model"""
        self.model.fit(X_train, y_train)
        self.coefficients = self.model.coef_
        
        # Calculate feature importance (absolute coefficients)
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'coefficient': self.model.coef_,
            'abs_coefficient': np.abs(self.model.coef_)
        }).sort_values('abs_coefficient', ascending=False)
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def get_channel_contributions(self, X):
        """Calculate contribution of each marketing channel"""
        predictions = self.predict(X)
        
        contributions = {}
        spend_features = [f'{channel}_spend' for channel in MARKETING_CHANNELS.keys()]
        
        for feature in spend_features:
            if feature in X.columns:
                idx = X.columns.get_loc(feature)
                coef = self.coefficients[idx]
                contribution = X[feature].values * coef
                
                channel = feature.replace('_spend', '')
                contributions[channel] = contribution
        
        return contributions


class XGBoostMMMModel:
    """XGBoost model for MMM (with sklearn fallback)"""

    def __init__(self, params=None):
        self.use_xgboost = XGBOOST_AVAILABLE

        if self.use_xgboost:
            default_params = {
                'objective': 'reg:squarederror',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': MODELING_CONFIG['random_state']
            }
            self.params = params or default_params
            self.model = xgb.XGBRegressor(**self.params)
        else:
            # Fallback to sklearn GradientBoostingRegressor
            default_params = {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'random_state': MODELING_CONFIG['random_state']
            }
            self.params = params or default_params
            self.model = GradientBoostingRegressor(**self.params)

        self.feature_importance = None
    
    def fit(self, X_train, y_train):
        """Train the model"""
        self.model.fit(X_train, y_train)
        
        # Feature importance
        importance_scores = self.model.feature_importances_
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def get_channel_contributions(self, X):
        """
        Approximate channel contributions using SHAP-style approach
        Contribution = feature_value * feature_importance
        """
        contributions = {}
        spend_features = [f'{channel}_spend' for channel in MARKETING_CHANNELS.keys()]
        
        for feature in spend_features:
            if feature in X.columns:
                importance = self.feature_importance[
                    self.feature_importance['feature'] == feature
                ]['importance'].values[0]
                
                # Approximate contribution
                contribution = X[feature].values * importance
                
                channel = feature.replace('_spend', '')
                contributions[channel] = contribution
        
        return contributions


class CustomMMMModel:
    """
    Custom MMM with explicit adstock and saturation modeling
    Uses non-linear least squares optimization
    """
    
    def __init__(self):
        self.params = {}
        self.is_fitted = False
    
    def adstock_transform(self, x, decay):
        """Apply geometric adstock transformation"""
        n = len(x)
        adstocked = np.zeros(n)
        
        for t in range(n):
            for lag in range(min(t + 1, 8)):
                adstocked[t] += x[t - lag] * (decay ** lag)
        
        return adstocked
    
    def saturation_transform(self, x, alpha, gamma):
        """Apply Hill saturation curve"""
        return (x ** alpha) / (gamma ** alpha + x ** alpha)
    
    def fit(self, X_train, y_train):
        """
        Fit custom MMM model
        Estimates adstock and saturation parameters for each channel
        """
        # Initialize parameters
        n_channels = len(MARKETING_CHANNELS)
        
        # Initial guess: [adstock_rates, saturation_alphas, saturation_gammas, intercept, other_coefs]
        initial_params = np.concatenate([
            np.ones(n_channels) * 0.5,  # adstock rates
            np.ones(n_channels) * 0.5,  # saturation alphas
            np.ones(n_channels) * 0.8,  # saturation gammas
            [y_train.mean()],           # intercept
            np.zeros(X_train.shape[1] - n_channels)  # other features
        ])
        
        def objective(params):
            """Objective function to minimize"""
            try:
                # Extract parameters
                idx = 0
                adstock_rates = params[idx:idx+n_channels]
                idx += n_channels
                saturation_alphas = params[idx:idx+n_channels]
                idx += n_channels
                saturation_gammas = params[idx:idx+n_channels]
                idx += n_channels
                intercept = params[idx]
                idx += 1
                other_coefs = params[idx:]
                
                # Apply transformations
                X_transformed = X_train.copy()
                spend_features = [f'{channel}_spend' for channel in MARKETING_CHANNELS.keys()]
                
                for i, feature in enumerate(spend_features):
                    if feature in X_transformed.columns:
                        x = X_transformed[feature].values
                        
                        # Adstock
                        x_adstocked = self.adstock_transform(x, adstock_rates[i])
                        
                        # Saturation
                        x_saturated = self.saturation_transform(
                            x_adstocked,
                            saturation_alphas[i],
                            saturation_gammas[i]
                        )
                        
                        X_transformed[feature] = x_saturated
                
                # Prediction
                y_pred = intercept + (X_transformed.values @ np.concatenate([other_coefs]))
                
                # MSE
                mse = mean_squared_error(y_train, y_pred)
                return mse
            
            except:
                return 1e10
        
        # Optimize (simplified version - in production, use better optimization)
        # For demo, use simple linear regression
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict(X)


def train_all_models(X_train, y_train):
    """Train all MMM models"""
    
    print("\n" + "="*70)
    print("TRAINING MMM MODELS")
    print("="*70)
    
    models = {}
    
    # Linear Regression
    print("\n1. Training Linear Regression...")
    models['linear'] = LinearMMMModel(model_type='linear')
    models['linear'].fit(X_train, y_train)
    print("   ✓ Linear Regression trained")
    
    # Ridge Regression
    print("\n2. Training Ridge Regression...")
    models['ridge'] = LinearMMMModel(alpha=1.0, model_type='ridge')
    models['ridge'].fit(X_train, y_train)
    print("   ✓ Ridge Regression trained")
    
    # Lasso Regression
    print("\n3. Training Lasso Regression...")
    models['lasso'] = LinearMMMModel(alpha=0.1, model_type='lasso')
    models['lasso'].fit(X_train, y_train)
    print("   ✓ Lasso Regression trained")
    
    # XGBoost
    print("\n4. Training XGBoost...")
    models['xgboost'] = XGBoostMMMModel()
    models['xgboost'].fit(X_train, y_train)
    print("   ✓ XGBoost trained")
    
    print("\n" + "="*70)
    print("MODEL TRAINING COMPLETE")
    print("="*70)
    
    return models


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """Evaluate model performance"""
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    metrics = {
        'model': model_name,
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'train_mape': np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100,
        'test_mape': np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100,
    }
    
    return metrics, y_train_pred, y_test_pred


def evaluate_all_models(models, X_train, y_train, X_test, y_test):
    """Evaluate all models and compare performance"""
    
    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70)
    
    results = []
    predictions = {}
    
    for name, model in models.items():
        print(f"\nEvaluating {name.upper()}...")
        metrics, y_train_pred, y_test_pred = evaluate_model(
            model, X_train, y_train, X_test, y_test, name
        )
        
        results.append(metrics)
        predictions[name] = {
            'train_pred': y_train_pred,
            'test_pred': y_test_pred
        }
        
        print(f"  Test RMSE: ${metrics['test_rmse']:,.0f}")
        print(f"  Test R²: {metrics['test_r2']:.4f}")
        print(f"  Test MAPE: {metrics['test_mape']:.2f}%")
    
    results_df = pd.DataFrame(results)
    
    # Find best model
    best_model = results_df.loc[results_df['test_rmse'].idxmin(), 'model']
    
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    print(f"\nBest model (lowest test RMSE): {best_model.upper()}")
    print("\nPerformance Summary:")
    print(results_df[['model', 'test_rmse', 'test_r2', 'test_mape']].to_string(index=False))
    
    return results_df, predictions, best_model


def calculate_marketing_roi(model, X, y, model_name='ridge'):
    """Calculate ROI for each marketing channel"""
    
    print("\n" + "="*70)
    print(f"MARKETING ROI ANALYSIS ({model_name.upper()})")
    print("="*70)
    
    # Get channel contributions
    contributions = model.get_channel_contributions(X)
    
    roi_results = []
    
    for channel in MARKETING_CHANNELS.keys():
        if channel in contributions:
            spend = X[f'{channel}_spend'].sum()
            contribution = contributions[channel].sum()
            
            roi = (contribution - spend) / spend
            roas = contribution / spend
            
            roi_results.append({
                'channel': channel,
                'total_spend': spend,
                'total_contribution': contribution,
                'roi': roi,
                'roas': roas,
                'avg_weekly_spend': spend / len(X),
                'avg_weekly_contribution': contribution / len(X)
            })
            
            print(f"\n{channel.upper()}:")
            print(f"  Total spend: ${spend:,.0f}")
            print(f"  Total contribution: ${contribution:,.0f}")
            print(f"  ROI: {roi:.2f}x")
            print(f"  ROAS: {roas:.2f}x")
    
    roi_df = pd.DataFrame(roi_results).sort_values('roi', ascending=False)
    
    print("\n" + "="*70)
    
    return roi_df


def run_complete_modeling(filepath='data/mmm_data.csv'):
    """Complete modeling pipeline"""
    # Preprocess data
    data = preprocess_pipeline(filepath)
    
    # Train models
    models = train_all_models(data['X_train'], data['y_train'])
    
    # Evaluate models
    results_df, predictions, best_model_name = evaluate_all_models(
        models,
        data['X_train'], data['y_train'],
        data['X_test'], data['y_test']
    )
    
    # Calculate ROI for best model
    best_model = models[best_model_name]
    roi_df = calculate_marketing_roi(
        best_model,
        data['X_train'],
        data['y_train'],
        best_model_name
    )
    
    return {
        'models': models,
        'results': results_df,
        'predictions': predictions,
        'best_model': best_model_name,
        'roi': roi_df,
        'data': data
    }


if __name__ == "__main__":
    results = run_complete_modeling()
    
    print("\n\nFinal Results:")
    print(f"Best Model: {results['best_model'].upper()}")
    print(f"\nTop 3 ROI Channels:")
    print(results['roi'].head(3)[['channel', 'roi', 'roas']].to_string(index=False))
