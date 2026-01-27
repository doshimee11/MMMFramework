# Marketing Mix Modeling (MMM) Framework

A comprehensive Python framework for Marketing Mix Modeling with support for adstock transformations, saturation curves, and budget optimization. Features an interactive Streamlit dashboard for analysis and scenario planning.

## Features

- **Multiple Regression Models**: Linear, Ridge, Lasso, and XGBoost regressors
- **Adstock Transformations**: Geometric, Weibull, and delayed decay functions
- **Saturation Curves**: Hill function for diminishing returns modeling
- **Budget Optimization**: Constrained optimization for marketing spend allocation
- **Channel Analysis**: ROI and ROAS calculations per marketing channel
- **Scenario Planning**: Compare current vs. optimized budget allocations
- **Interactive Dashboard**: Streamlit-based UI for exploration and visualization

## Project Structure

```
MMMFramework/
├── config.py                 # Configuration parameters
├── streamlit_app.py          # Interactive Streamlit dashboard
├── requirements.txt          # Python dependencies
├── data/                     # Sample datasets
└── src/
    ├── data_generation.py    # Synthetic marketing data generation
    ├── data_processing.py    # Adstock and saturation transformations
    ├── data_modeling.py      # Regression models (Linear, Ridge, XGBoost)
    └── optimization.py       # Budget allocation optimization
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/MMMFramework.git
cd MMMFramework
```

2. Create and activate a virtual environment:
```bash
python3 -m venv mmm_venv
source mmm_venv/bin/activate  # On Windows: mmm_venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

### Configuration

Modify `config.py` to customize your marketing channels:

```python
MARKETING_CHANNELS = {
    'tv': {
        'name': 'TV Advertising',
        'weekly_budget_mean': 50000,
        'base_effectiveness': 0.15,
        'adstock_rate': 0.4,
        'saturation_alpha': 0.5,
        'saturation_gamma': 0.8,
    },
    'digital': {
        'name': 'Digital Advertising',
        'weekly_budget_mean': 35000,
        'base_effectiveness': 0.20,
        'adstock_rate': 0.3,
        ...
    },
}
```

## Dashboard Features

1. **Data Overview**: Visualize marketing spend and sales over time
2. **Model Performance**: Compare Linear, Ridge, Lasso, and XGBoost models
3. **Channel Contribution**: Decompose sales by marketing channel
4. **ROI Analysis**: Calculate return on investment per channel
5. **Budget Optimization**: Find optimal spend allocation
6. **Scenario Planning**: Test different budget scenarios

## Supported Marketing Channels

- TV Advertising
- Digital Advertising
- Social Media
- Search Marketing
- Print Advertising
- Email Marketing

## Key Concepts

### Adstock Transformation
Models the carryover effect of advertising over time:
```
Adstock(t) = Spend(t) + decay_rate * Adstock(t-1)
```

### Saturation (Hill Function)
Models diminishing returns at higher spend levels:
```
Saturation(x) = x^alpha / (x^alpha + gamma^alpha)
```

### Budget Optimization
Maximize sales subject to budget constraints using scipy optimization.

## Dependencies

- streamlit
- pandas
- numpy
- scipy
- scikit-learn
- xgboost (optional, with sklearn fallback)
- statsmodels
- plotly
- seaborn

## License

MIT License
