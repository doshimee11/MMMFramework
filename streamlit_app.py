"""
Marketing Mix Modeling Dashboard
Interactive Streamlit application for MMM analysis
"""

import os
import sys
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data_processing import load_data
from src.data_modeling import run_complete_modeling
from src.optimization import compare_scenarios, BudgetOptimizer
from config import MARKETING_CHANNELS


# Page config
st.set_page_config(
    page_title="📊 Marketing Mix Modeling",
    page_icon="📊",
    layout="wide"
)


# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #1f77b4;}
    .metric-card {background-color: #f0f2f6; padding: 15px; border-radius: 5px;}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_mmm_data():
    """Load MMM data"""
    return load_data('data/mmm_data.csv')


@st.cache_resource
def train_mmm_models():
    """Train MMM models (cached)"""
    return run_complete_modeling()


def main():
    # Title
    st.markdown('<p class="main-header">📊 Marketing Mix Modeling Dashboard</p>', 
                unsafe_allow_html=True)
    st.markdown("**Optimize your marketing budget allocation with data-driven insights**")
    st.markdown("---")
    
    # Load data
    try:
        df = load_mmm_data()
        mmm_results = train_mmm_models()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Please run: `python src/data_generation.py` first")
        return
    
    # Sidebar
    st.sidebar.header("📋 Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Overview", "Model Performance", "Channel Analysis", "Budget Optimization", "Scenario Planning"]
    )
    
    # ==================================================================
    # OVERVIEW PAGE
    # ==================================================================
    
    if page == "Overview":
        st.header("📈 Executive Summary")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Sales",
                f"${df['total_sales'].sum()/1e6:.1f}M",
                help="Total sales over analysis period"
            )
        
        with col2:
            st.metric(
                "Marketing Spend",
                f"${df['total_marketing_spend'].sum()/1e6:.1f}M",
                help="Total marketing investment"
            )
        
        with col3:
            st.metric(
                "Average ROI",
                f"{df['total_marketing_roi'].mean():.2f}x",
                help="Average return on marketing investment"
            )
        
        with col4:
            st.metric(
                "Best Model",
                mmm_results['best_model'].upper(),
                help="Model with lowest test error"
            )
        
        st.markdown("---")
        
        # Sales over time
        st.subheader("Sales Trend")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['total_sales'],
            mode='lines',
            name='Total Sales',
            line=dict(color='#636EFA', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['base_sales'] + df['trend'] + df['seasonality'],
            mode='lines',
            name='Baseline (No Marketing)',
            line=dict(color='#EF553B', width=2, dash='dash')
        ))
        
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Sales ($)',
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, width='stretch')
        
        # Marketing contribution
        st.subheader("Marketing Contribution Over Time")
        
        fig = go.Figure()
        
        channels = list(MARKETING_CHANNELS.keys())
        colors = px.colors.qualitative.Set2
        
        for i, channel in enumerate(channels):
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df[f'{channel}_contribution'],
                mode='lines',
                name=channel.upper(),
                stackgroup='one',
                line=dict(color=colors[i % len(colors)])
            ))
        
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Contribution to Sales ($)',
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, width='stretch')
    
    # ==================================================================
    # MODEL PERFORMANCE PAGE
    # ==================================================================
    
    elif page == "Model Performance":
        st.header("🎯 Model Performance")
        
        # Model comparison
        st.subheader("Model Comparison")
        
        results_df = mmm_results['results']
        
        # Display table
        display_df = results_df[['model', 'test_rmse', 'test_r2', 'test_mape']].copy()
        display_df.columns = ['Model', 'Test RMSE ($)', 'Test R²', 'Test MAPE (%)']
        display_df['Test RMSE ($)'] = display_df['Test RMSE ($)'].apply(lambda x: f"${x:,.0f}")
        display_df['Test R²'] = display_df['Test R²'].apply(lambda x: f"{x:.4f}")
        display_df['Test MAPE (%)'] = display_df['Test MAPE (%)'].apply(lambda x: f"{x:.2f}%")
        
        st.dataframe(display_df, width='stretch')
        
        # Model comparison chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=results_df['model'],
            y=results_df['test_r2'],
            name='Test R²',
            marker_color='#636EFA'
        ))
        
        fig.update_layout(
            title='Model Performance (Test R²)',
            xaxis_title='Model',
            yaxis_title='R² Score',
            height=400
        )
        
        st.plotly_chart(fig, width='stretch')
        
        # Actual vs Predicted
        st.subheader("Actual vs Predicted Sales")
        
        best_model_name = mmm_results['best_model']
        predictions = mmm_results['predictions'][best_model_name]
        
        # Get actual values
        y_test = mmm_results['data']['y_test']
        y_test_pred = predictions['test_pred']
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=y_test,
            y=y_test_pred,
            mode='markers',
            name='Predictions',
            marker=dict(size=8, color='#636EFA', opacity=0.6)
        ))
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_test_pred.min())
        max_val = max(y_test.max(), y_test_pred.max())
        
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title=f'Actual vs Predicted Sales ({best_model_name.upper()})',
            xaxis_title='Actual Sales ($)',
            yaxis_title='Predicted Sales ($)',
            height=500
        )
        
        st.plotly_chart(fig, width='stretch')
        
        # Feature importance
        st.subheader("Feature Importance")
        
        best_model = mmm_results['models'][best_model_name]
        
        if hasattr(best_model, 'feature_importance'):
            importance_df = best_model.feature_importance.head(15)
            
            fig = go.Figure()
            
            if 'importance' in importance_df.columns:
                fig.add_trace(go.Bar(
                    y=importance_df['feature'],
                    x=importance_df['importance'],
                    orientation='h',
                    marker_color='#636EFA'
                ))
                x_title = 'Importance Score'
            else:
                fig.add_trace(go.Bar(
                    y=importance_df['feature'],
                    x=importance_df['abs_coefficient'],
                    orientation='h',
                    marker_color='#636EFA'
                ))
                x_title = 'Absolute Coefficient'
            
            fig.update_layout(
                title='Top 15 Most Important Features',
                xaxis_title=x_title,
                yaxis_title='Feature',
                height=500
            )
            
            st.plotly_chart(fig, width='stretch')
    
    # ==================================================================
    # CHANNEL ANALYSIS PAGE
    # ==================================================================
    
    elif page == "Channel Analysis":
        st.header("📺 Channel Performance Analysis")
        
        # Channel ROI
        roi_df = mmm_results['roi']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("ROI by Channel")
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=roi_df['channel'],
                y=roi_df['roi'],
                marker_color='#636EFA',
                text=[f"{x:.2f}x" for x in roi_df['roi']],
                textposition='auto'
            ))
            
            fig.update_layout(
                xaxis_title='Channel',
                yaxis_title='ROI (x)',
                height=400
            )
            
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            st.subheader("ROAS by Channel")
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=roi_df['channel'],
                y=roi_df['roas'],
                marker_color='#EF553B',
                text=[f"{x:.2f}x" for x in roi_df['roas']],
                textposition='auto'
            ))
            
            fig.update_layout(
                xaxis_title='Channel',
                yaxis_title='ROAS (x)',
                height=400
            )
            
            st.plotly_chart(fig, width='stretch')
        
        # Channel contribution
        st.subheader("Total Contribution by Channel")
        
        fig = go.Figure()
        
        fig.add_trace(go.Pie(
            labels=[c.upper() for c in roi_df['channel']],
            values=roi_df['total_contribution'],
            hole=0.4
        ))
        
        fig.update_layout(
            title='Share of Total Marketing Contribution',
            height=500
        )
        
        st.plotly_chart(fig, width='stretch')
        
        # Detailed metrics table
        st.subheader("Detailed Channel Metrics")
        
        display_roi = roi_df.copy()
        display_roi['total_spend'] = display_roi['total_spend'].apply(lambda x: f"${x:,.0f}")
        display_roi['total_contribution'] = display_roi['total_contribution'].apply(lambda x: f"${x:,.0f}")
        display_roi['roi'] = display_roi['roi'].apply(lambda x: f"{x:.2f}x")
        display_roi['roas'] = display_roi['roas'].apply(lambda x: f"{x:.2f}x")
        display_roi['avg_weekly_spend'] = display_roi['avg_weekly_spend'].apply(lambda x: f"${x:,.0f}")
        display_roi['avg_weekly_contribution'] = display_roi['avg_weekly_contribution'].apply(lambda x: f"${x:,.0f}")
        
        st.dataframe(display_roi, width='stretch')
    
    # ==================================================================
    # BUDGET OPTIMIZATION PAGE
    # ==================================================================
    
    elif page == "Budget Optimization":
        st.header("💰 Budget Optimization")
        
        st.markdown("""
        Find the optimal allocation of your marketing budget across channels to maximize sales or ROI.
        """)
        
        # Input parameters
        col1, col2 = st.columns(2)
        
        with col1:
            total_budget = st.number_input(
                "Total Weekly Budget ($)",
                min_value=50000,
                max_value=500000,
                value=180000,
                step=10000
            )
        
        with col2:
            objective = st.selectbox(
                "Optimization Objective",
                ["Maximize Sales", "Maximize ROI"]
            )
        
        if st.button("🔍 Optimize Budget", type="primary"):
            with st.spinner("Optimizing budget allocation..."):
                # Get model and data
                best_model = mmm_results['models'][mmm_results['best_model']]
                X_train = mmm_results['data']['X_train']
                feature_columns = mmm_results['data']['feature_columns']
                
                # Create optimizer
                X_mean = X_train.mean().values
                optimizer = BudgetOptimizer(best_model, X_mean, feature_columns)
                
                # Run optimization
                obj_type = 'maximize_sales' if objective == "Maximize Sales" else 'maximize_roi'
                optimal_result = optimizer.optimize(total_budget=total_budget, objective=obj_type)
                
                # Display results
                st.success("✅ Optimization Complete!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Predicted Sales",
                        f"${optimal_result['predicted_sales']:,.0f}"
                    )
                
                with col2:
                    st.metric(
                        "Total Spend",
                        f"${optimal_result['total_budget']:,.0f}"
                    )
                
                with col3:
                    st.metric(
                        "Expected ROI",
                        f"{optimal_result['total_roi']:.2f}x"
                    )
                
                # Allocation breakdown
                st.subheader("Optimal Allocation")
                
                allocation_data = []
                for channel, alloc in optimal_result['allocation'].items():
                    allocation_data.append({
                        'Channel': channel.upper(),
                        'Budget': alloc['spend'],
                        'Percentage': alloc['percentage']
                    })
                
                allocation_df = pd.DataFrame(allocation_data).sort_values('Budget', ascending=False)
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=allocation_df['Channel'],
                    y=allocation_df['Budget'],
                    marker_color='#636EFA',
                    text=[f"${x:,.0f}<br>({allocation_df.loc[i, 'Percentage']:.1f}%)" 
                          for i, x in enumerate(allocation_df['Budget'])],
                    textposition='auto'
                ))
                
                fig.update_layout(
                    title='Optimal Budget Allocation by Channel',
                    xaxis_title='Channel',
                    yaxis_title='Weekly Budget ($)',
                    height=500
                )
                
                st.plotly_chart(fig, width='stretch')
    
    # ==================================================================
    # SCENARIO PLANNING PAGE
    # ==================================================================
    
    elif page == "Scenario Planning":
        st.header("🎯 Scenario Planning")
        
        st.markdown("""
        Compare different budget allocation strategies to understand their impact on sales and ROI.
        """)
        
        if st.button("📊 Run Scenario Analysis", type="primary"):
            with st.spinner("Analyzing scenarios..."):
                # Get model and data
                best_model = mmm_results['models'][mmm_results['best_model']]
                X_train = mmm_results['data']['X_train']
                feature_columns = mmm_results['data']['feature_columns']
                
                # Run scenario comparison
                scenarios = compare_scenarios(best_model, X_train, feature_columns)
                
                # Display comparison
                st.subheader("Scenario Comparison")
                
                comparison_data = []
                for scenario_name, scenario_data in scenarios.items():
                    comparison_data.append({
                        'Scenario': scenario_name.upper(),
                        'Total Spend': f"${scenario_data['total_budget']:,.0f}",
                        'Predicted Sales': f"${scenario_data['predicted_sales']:,.0f}",
                        'ROI': f"{scenario_data['total_roi']:.2f}x",
                        'Sales Lift': f"{(scenario_data['predicted_sales'] / scenarios['current']['predicted_sales'] - 1) * 100:+.1f}%"
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, width='stretch')
                
                # Allocation comparison
                st.subheader("Allocation Breakdown by Scenario")
                
                scenario_names = list(scenarios.keys())
                selected_scenario = st.selectbox("Select Scenario", scenario_names)
                
                if selected_scenario:
                    scenario_data = scenarios[selected_scenario]
                    
                    allocation_data = []
                    for channel, alloc in scenario_data['allocation'].items():
                        allocation_data.append({
                            'Channel': channel.upper(),
                            'Spend': alloc['spend'],
                            'Percentage': alloc['percentage']
                        })
                    
                    allocation_df = pd.DataFrame(allocation_data).sort_values('Spend', ascending=False)
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        x=allocation_df['Channel'],
                        y=allocation_df['Spend'],
                        marker_color='#636EFA',
                        text=[f"${x:,.0f}<br>({allocation_df.loc[i, 'Percentage']:.1f}%)" 
                              for i, x in enumerate(allocation_df['Spend'])],
                        textposition='auto'
                    ))
                    
                    fig.update_layout(
                        title=f'Budget Allocation - {selected_scenario.upper()}',
                        xaxis_title='Channel',
                        yaxis_title='Weekly Budget ($)',
                        height=500
                    )
                    
                    st.plotly_chart(fig, width='stretch')


if __name__ == "__main__":
    main()
