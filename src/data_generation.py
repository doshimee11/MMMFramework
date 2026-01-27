"""
Marketing Mix Modeling - Data Generation
Generates realistic marketing and sales data with:
- Adstock effects (carryover)
- Saturation curves (diminishing returns)
- Seasonality
- External factors (holidays, competitors, promotions)
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import weibull_min

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BUSINESS_CONFIG, MARKETING_CHANNELS, SALES_CONFIG,
    EXTERNAL_FACTORS, ADSTOCK_CONFIG, SATURATION_CONFIG
)

np.random.seed(42)


def generate_date_range():
    """Generate weekly date range"""
    start = pd.to_datetime(BUSINESS_CONFIG['start_date'])
    end = pd.to_datetime(BUSINESS_CONFIG['end_date'])
    
    dates = pd.date_range(start=start, end=end, freq=BUSINESS_CONFIG['frequency'])
    return dates


def generate_marketing_spend(dates):
    """Generate realistic marketing spend for each channel"""
    n_periods = len(dates)
    spend_data = {'date': dates}
    
    for channel, params in MARKETING_CHANNELS.items():
        # Base spend with random variation
        base_spend = np.random.normal(
            params['weekly_budget_mean'],
            params['weekly_budget_std'],
            n_periods
        )
        
        # Add trend (gradual increase over time)
        trend = np.linspace(0, params['weekly_budget_mean'] * 0.1, n_periods)
        
        # Add seasonality
        seasonality = params['weekly_budget_mean'] * 0.1 * np.sin(
            2 * np.pi * np.arange(n_periods) / 52
        )
        
        # Combine and ensure non-negative
        spend = base_spend + trend + seasonality
        spend = np.maximum(spend, 0)
        
        spend_data[f'{channel}_spend'] = spend
    
    return pd.DataFrame(spend_data)


def apply_adstock(spend, decay_rate, adstock_type='geometric', max_lag=8):
    """
    Apply adstock transformation to capture carryover effects
    
    Types:
    - geometric: Simple exponential decay
    - weibull: S-shaped decay (delayed then rapid)
    - delayed: Effect starts after delay period
    """
    n = len(spend)
    adstocked = np.zeros(n)
    
    if adstock_type == 'geometric':
        # Geometric adstock: X_t = x_t + decay * X_{t-1}
        for t in range(n):
            for lag in range(min(t + 1, max_lag)):
                adstocked[t] += spend[t - lag] * (decay_rate ** lag)
    
    elif adstock_type == 'weibull':
        # Weibull adstock (S-shaped decay)
        shape = ADSTOCK_CONFIG['weibull']['shape']
        scale = ADSTOCK_CONFIG['weibull']['scale']
        
        weights = weibull_min.pdf(np.arange(max_lag), shape, scale=scale)
        weights = weights / weights.sum()  # Normalize
        
        for t in range(n):
            for lag in range(min(t + 1, max_lag)):
                adstocked[t] += spend[t - lag] * weights[lag] * decay_rate
    
    elif adstock_type == 'delayed':
        # Delayed adstock (effect starts after delay)
        delay = ADSTOCK_CONFIG['delayed']['delay']
        theta = ADSTOCK_CONFIG['delayed']['theta']
        
        for t in range(n):
            for lag in range(delay, min(t + 1, max_lag)):
                adstocked[t] += spend[t - lag] * (theta ** (lag - delay))
    
    return adstocked


def apply_saturation(adstocked_spend, alpha=0.5, gamma=1.0, model='hill'):
    """
    Apply saturation curve (diminishing returns)
    
    Hill curve: S(x) = x^alpha / (gamma^alpha + x^alpha)
    - alpha: Shape parameter (elasticity)
    - gamma: Half-saturation point
    """
    if model == 'hill':
        # Hill saturation curve
        saturated = (adstocked_spend ** alpha) / (gamma ** alpha + adstocked_spend ** alpha)
    
    elif model == 'logistic':
        # Logistic saturation
        saturated = 1 / (1 + np.exp(-alpha * (adstocked_spend - gamma)))
    
    elif model == 'exponential':
        # Exponential saturation
        saturated = 1 - np.exp(-alpha * adstocked_spend / gamma)
    
    else:
        saturated = adstocked_spend
    
    return saturated


def generate_external_factors(dates):
    """Generate external factors (holidays, competitors, promotions)"""
    n_periods = len(dates)
    
    # Extract week numbers
    dates_dt = pd.to_datetime(dates)
    week_numbers = dates_dt.dt.isocalendar().week.values
    # week_numbers = pd.to_datetime(dates).isocalendar().week
    
    # Holidays
    holiday_weeks = EXTERNAL_FACTORS['holiday']['weeks']
    is_holiday = np.isin(week_numbers, holiday_weeks).astype(float)
    holiday_effect = is_holiday * EXTERNAL_FACTORS['holiday']['effect']
    
    # Competitor activity
    competitor_prob = EXTERNAL_FACTORS['competitor_activity']['probability']
    competitor_active = np.random.binomial(1, competitor_prob, n_periods)
    competitor_effect = competitor_active * EXTERNAL_FACTORS['competitor_activity']['effect']
    
    # Price promotions
    promotion_prob = EXTERNAL_FACTORS['price_promotions']['probability']
    has_promotion = np.random.binomial(1, promotion_prob, n_periods)
    promotion_effect = has_promotion * EXTERNAL_FACTORS['price_promotions']['effect']
    
    return {
        'is_holiday': is_holiday,
        'holiday_effect': holiday_effect,
        'competitor_active': competitor_active,
        'competitor_effect': competitor_effect,
        'has_promotion': has_promotion,
        'promotion_effect': promotion_effect,
    }


def generate_sales(spend_df):
    """Generate sales based on marketing spend and external factors"""
    n_periods = len(spend_df)
    
    # Base sales
    base_sales = SALES_CONFIG['base_sales']
    
    # Trend
    trend = SALES_CONFIG['trend_coefficient'] * np.arange(n_periods)
    
    # Seasonality
    seasonality = base_sales * SALES_CONFIG['seasonality_amplitude'] * np.sin(
        2 * np.pi * np.arange(n_periods) / 52
    )
    
    # External factors
    external = generate_external_factors(spend_df['date'])
    external_effect = (1 + external['holiday_effect'] + 
                      external['competitor_effect'] + 
                      external['promotion_effect'])
    
    # Marketing contribution
    marketing_contribution = np.zeros(n_periods)
    
    for channel, params in MARKETING_CHANNELS.items():
        # Get spend
        spend = spend_df[f'{channel}_spend'].values
        
        # Apply adstock
        adstocked = apply_adstock(
            spend,
            params['adstock_rate'],
            params['adstock_type'],
            ADSTOCK_CONFIG['geometric']['max_lag']
        )
        
        # Apply saturation
        saturated = apply_saturation(
            adstocked,
            params['saturation_alpha'],
            params['saturation_gamma'],
            SATURATION_CONFIG['model']
        )
        
        # Contribution to sales
        contribution = saturated * params['base_effectiveness'] * base_sales
        marketing_contribution += contribution
        
        # Store individual channel contributions
        spend_df[f'{channel}_contribution'] = contribution
    
    # Total sales
    sales = (base_sales + trend + seasonality + marketing_contribution) * external_effect
    
    # Add noise
    noise = np.random.normal(0, SALES_CONFIG['noise_std'], n_periods)
    sales = sales + noise
    
    # Ensure non-negative
    sales = np.maximum(sales, 0)
    
    # Add components to dataframe
    spend_df['base_sales'] = base_sales
    spend_df['trend'] = trend
    spend_df['seasonality'] = seasonality
    spend_df['marketing_contribution'] = marketing_contribution
    spend_df['total_sales'] = sales
    
    # Add external factors
    for key, value in external.items():
        spend_df[key] = value
    
    return spend_df


def calculate_roi_metrics(df):
    """Calculate ROI and ROAS for each channel"""
    
    for channel in MARKETING_CHANNELS.keys():
        spend = df[f'{channel}_spend']
        contribution = df[f'{channel}_contribution']
        
        # ROI = (Revenue - Cost) / Cost
        df[f'{channel}_roi'] = (contribution - spend) / spend
        
        # ROAS = Revenue / Cost
        df[f'{channel}_roas'] = contribution / spend
    
    # Total marketing metrics
    total_spend = sum(df[f'{channel}_spend'] for channel in MARKETING_CHANNELS.keys())
    total_contribution = df['marketing_contribution']
    
    df['total_marketing_spend'] = total_spend
    df['total_marketing_roi'] = (total_contribution - total_spend) / total_spend
    df['total_marketing_roas'] = total_contribution / total_spend
    
    return df


def generate_mmm_data():
    """Generate complete MMM dataset"""
    
    print("="*70)
    print("GENERATING MARKETING MIX MODELING DATA")
    print("="*70)
    print(f"\nBusiness: {BUSINESS_CONFIG['company_name']}")
    print(f"Period: {BUSINESS_CONFIG['start_date']} to {BUSINESS_CONFIG['end_date']}")
    print(f"Frequency: {BUSINESS_CONFIG['frequency']}")
    print(f"Channels: {len(MARKETING_CHANNELS)}")
    print()
    
    # Generate dates
    dates = generate_date_range()
    print(f"Generated {len(dates)} periods")
    
    # Generate marketing spend
    print("Generating marketing spend...")
    spend_df = generate_marketing_spend(dates)
    
    # Generate sales
    print("Generating sales with marketing effects...")
    df = generate_sales(spend_df)
    
    # Calculate ROI metrics
    print("Calculating ROI metrics...")
    df = calculate_roi_metrics(df)
    
    # Summary statistics
    print("\n" + "="*70)
    print("DATA GENERATION SUMMARY")
    print("="*70)
    print(f"\nTotal periods: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    print(f"\nSales Statistics:")
    print(f"  Average weekly sales: ${df['total_sales'].mean():,.0f}")
    print(f"  Total sales: ${df['total_sales'].sum():,.0f}")
    print(f"  Sales range: ${df['total_sales'].min():,.0f} - ${df['total_sales'].max():,.0f}")
    
    print(f"\nMarketing Statistics:")
    print(f"  Average weekly spend: ${df['total_marketing_spend'].mean():,.0f}")
    print(f"  Total marketing spend: ${df['total_marketing_spend'].sum():,.0f}")
    print(f"  Average marketing ROI: {df['total_marketing_roi'].mean():.2f}")
    print(f"  Average marketing ROAS: {df['total_marketing_roas'].mean():.2f}")
    
    print(f"\nChannel Performance:")
    for channel in MARKETING_CHANNELS.keys():
        avg_spend = df[f'{channel}_spend'].mean()
        avg_contribution = df[f'{channel}_contribution'].mean()
        avg_roi = df[f'{channel}_roi'].mean()
        avg_roas = df[f'{channel}_roas'].mean()
        
        print(f"  {channel.upper()}:")
        print(f"    Avg spend: ${avg_spend:,.0f}")
        print(f"    Avg contribution: ${avg_contribution:,.0f}")
        print(f"    Avg ROI: {avg_roi:.2f}")
        print(f"    Avg ROAS: {avg_roas:.2f}")
    
    # Save to CSV
    os.makedirs('data', exist_ok=True)
    output_path = 'data/mmm_data.csv'
    df.to_csv(output_path, index=False)
    
    print(f"\n✓ Data saved to {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024:.2f} KB")
    print("="*70 + "\n")
    
    return df


if __name__ == "__main__":
    df = generate_mmm_data()
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nColumn names:")
    print(df.columns.tolist())
