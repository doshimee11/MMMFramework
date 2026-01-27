"""
Marketing Mix Modeling (MMM) Configuration
Customize these parameters for your business
"""

import numpy as np


BUSINESS_CONFIG = {
    'company_name': 'RetailCo',
    'category': 'E-commerce',
    'start_date': '2022-01-01',
    'end_date': '2024-12-31',
    'frequency': 'W',  # Weekly data (W, M for monthly, D for daily)
}


# Define all marketing channels
MARKETING_CHANNELS = {
    'tv': {
        'name': 'TV Advertising',
        'weekly_budget_mean': 50000,
        'weekly_budget_std': 15000,
        'base_effectiveness': 0.15,      # Base contribution to sales
        'adstock_rate': 0.4,             # Carryover effect (0-1)
        'adstock_type': 'geometric',     # geometric, weibull, delayed
        'saturation_alpha': 0.5,         # Hill curve parameter
        'saturation_gamma': 0.8,         # Saturation threshold
    },
    'digital': {
        'name': 'Digital Advertising',
        'weekly_budget_mean': 35000,
        'weekly_budget_std': 10000,
        'base_effectiveness': 0.20,
        'adstock_rate': 0.3,
        'adstock_type': 'geometric',
        'saturation_alpha': 0.4,
        'saturation_gamma': 0.7,
    },
    'social': {
        'name': 'Social Media',
        'weekly_budget_mean': 25000,
        'weekly_budget_std': 8000,
        'base_effectiveness': 0.18,
        'adstock_rate': 0.5,
        'adstock_type': 'geometric',
        'saturation_alpha': 0.35,
        'saturation_gamma': 0.6,
    },
    'search': {
        'name': 'Search Marketing',
        'weekly_budget_mean': 40000,
        'weekly_budget_std': 12000,
        'base_effectiveness': 0.25,
        'adstock_rate': 0.2,
        'adstock_type': 'geometric',
        'saturation_alpha': 0.3,
        'saturation_gamma': 0.5,
    },
    'print': {
        'name': 'Print Advertising',
        'weekly_budget_mean': 20000,
        'weekly_budget_std': 6000,
        'base_effectiveness': 0.08,
        'adstock_rate': 0.6,
        'adstock_type': 'geometric',
        'saturation_alpha': 0.6,
        'saturation_gamma': 0.9,
    },
    'email': {
        'name': 'Email Marketing',
        'weekly_budget_mean': 10000,
        'weekly_budget_std': 3000,
        'base_effectiveness': 0.12,
        'adstock_rate': 0.1,
        'adstock_type': 'geometric',
        'saturation_alpha': 0.25,
        'saturation_gamma': 0.4,
    },
}

SALES_CONFIG = {
    'base_sales': 500000,           # Weekly baseline sales without marketing
    'trend_coefficient': 500,        # Weekly trend growth
    'seasonality_amplitude': 0.15,   # Seasonal variation (15%)
    'noise_std': 20000,             # Random noise standard deviation
}

EXTERNAL_FACTORS = {
    'holiday': {
        'effect': 0.25,              # 25% boost during holidays
        'weeks': [51, 52, 1, 2, 25, 48, 49, 50],  # Holiday weeks
    },
    'competitor_activity': {
        'effect': -0.10,             # -10% impact when competitor is active
        'probability': 0.2,          # 20% of weeks have competitor activity
    },
    'price_promotions': {
        'effect': 0.20,              # 20% boost during promotions
        'probability': 0.15,         # 15% of weeks have promotions
    },
}

ADSTOCK_CONFIG = {
    'geometric': {
        'max_lag': 8,                # Maximum weeks of carryover
    },
    'weibull': {
        'shape': 2.0,
        'scale': 3.0,
        'max_lag': 8,
    },
    'delayed': {
        'delay': 2,                  # Weeks before effect starts
        'theta': 0.5,
        'max_lag': 8,
    },
}

SATURATION_CONFIG = {
    'model': 'hill',                 # hill, logistic, exponential
    'hill_alpha': 0.5,               # Default alpha (if not channel-specific)
    'hill_gamma': 0.8,               # Default gamma (if not channel-specific)
}

MODELING_CONFIG = {
    'train_test_split': 0.8,         # 80% train, 20% test
    'validation_method': 'time_series',  # time_series, cross_validation
    'cv_folds': 5,
    'random_state': 42,
}

OPTIMIZATION_CONFIG = {
    'total_budget': 180000,          # Weekly total marketing budget
    'min_budget_per_channel': 5000,  # Minimum spend per channel
    'max_budget_per_channel': 80000, # Maximum spend per channel
    'objective': 'maximize_sales',   # maximize_sales, maximize_roi, minimize_cost
    'constraints': 'budget',         # budget, roi_threshold, both
    'min_roi_threshold': 2.0,        # Minimum acceptable ROI
}

SCENARIOS = {
    'current': {
        'name': 'Current Allocation',
        'description': 'Maintain current spending levels',
    },
    'optimized': {
        'name': 'Optimized Allocation',
        'description': 'Maximize sales given budget constraint',
    },
    'budget_cut_20': {
        'name': '20% Budget Cut',
        'description': 'Reduce total budget by 20%',
        'total_budget': 144000,
    },
    'budget_increase_20': {
        'name': '20% Budget Increase',
        'description': 'Increase total budget by 20%',
        'total_budget': 216000,
    },
    'digital_first': {
        'name': 'Digital First Strategy',
        'description': 'Focus 70% of budget on digital channels',
        'channel_constraints': {
            'digital': {'min': 0.30, 'max': 0.40},
            'social': {'min': 0.15, 'max': 0.25},
            'search': {'min': 0.20, 'max': 0.30},
        }
    },
}

DATABASE_CONFIG = {
    'database_url': 'sqlite:///data/mmm.db',
}

VIZ_CONFIG = {
    'color_palette': 'Set2',
    'figure_size': (12, 6),
    'dpi': 100,
}