"""
ARR Risk Attribution Dashboard
==============================
A production-grade Streamlit dashboard for customer churn risk attribution and root-cause analysis.
Designed with a dark theme, modern SaaS aesthetics, and interactive data exploration.

Author: Claude (Anthropic)
Version: 1.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

class Region(Enum):
    """Customer regions for filtering."""
    ALL = "All Regions"
    AMER = "AMER"
    EMEA = "EMEA"
    APAC = "APAC"

@dataclass
class Benchmark:
    """Benchmark configuration for risk metrics."""
    core_module_adoption: float = 0.80
    onboarding_completion: float = 0.90
    weekly_logins: int = 5
    time_to_first_value_days: int = 14
    seat_utilization: float = 0.75
    support_tickets_threshold: int = 3
    nps_score: float = 8.0

BENCHMARKS = Benchmark()

# Risk category weights (sum to 1.0)
RISK_WEIGHTS = {
    "product": 0.30,
    "process": 0.25,
    "development": 0.25,
    "relationship": 0.20
}

# Color palette
COLORS = {
    "background": "#0e1117",
    "card_bg": "#1a1d24",
    "card_border": "#2d3139",
    "text_primary": "#ffffff",
    "text_secondary": "#8b949e",
    "accent_red": "#ff4d4d",
    "accent_orange": "#ff9f43",
    "accent_yellow": "#feca57",
    "accent_green": "#26de81",
    "accent_blue": "#54a0ff",
    "accent_purple": "#a55eea",
}

# =============================================================================
# CUSTOM CSS STYLING
# =============================================================================

def inject_custom_css():
    """Inject custom CSS for dark theme and card styling."""
    st.markdown("""
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Outfit:wght@300;400;500;600;700&display=swap');
        
        /* Global styles */
        .stApp {
            background-color: #0e1117;
            font-family: 'Outfit', sans-serif;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #0a0c10;
            border-right: 1px solid #1f2937;
        }
        
        [data-testid="stSidebar"] .stMarkdown {
            color: #e5e7eb;
        }
        
        /* Custom card component */
        .risk-card {
            background: linear-gradient(145deg, #1a1d24 0%, #141720 100%);
            border: 1px solid #2d3139;
            border-radius: 16px;
            padding: 20px;
            margin: 8px 0;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
            transition: all 0.3s ease;
        }
        
        .risk-card:hover {
            border-color: #3d4149;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.5);
            transform: translateY(-2px);
        }
        
        .risk-card-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 16px;
        }
        
        .risk-card-title {
            font-family: 'Outfit', sans-serif;
            font-size: 1.1rem;
            font-weight: 600;
            color: #ffffff;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .risk-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
        }
        
        .risk-badge-high {
            background: rgba(255, 77, 77, 0.15);
            color: #ff4d4d;
            border: 1px solid rgba(255, 77, 77, 0.3);
        }
        
        .risk-badge-medium {
            background: rgba(255, 159, 67, 0.15);
            color: #ff9f43;
            border: 1px solid rgba(255, 159, 67, 0.3);
        }
        
        .risk-badge-low {
            background: rgba(38, 222, 129, 0.15);
            color: #26de81;
            border: 1px solid rgba(38, 222, 129, 0.3);
        }
        
        .metric-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 2rem;
            font-weight: 700;
            color: #ffffff;
            line-height: 1.2;
        }
        
        .metric-label {
            font-size: 0.85rem;
            color: #8b949e;
            margin-top: 4px;
        }
        
        .metric-delta {
            font-size: 0.9rem;
            font-weight: 500;
            margin-left: 8px;
        }
        
        .metric-delta-negative {
            color: #ff4d4d;
        }
        
        .metric-delta-positive {
            color: #26de81;
        }
        
        .benchmark-text {
            font-size: 0.8rem;
            color: #6b7280;
            margin-top: 8px;
        }
        
        .score-dots {
            display: flex;
            gap: 4px;
            margin-top: 12px;
        }
        
        .score-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #2d3139;
        }
        
        .score-dot-filled {
            background: #ff9f43;
        }
        
        .score-dot-green {
            background: #26de81;
        }
        
        /* Sub-card for nested metrics */
        .sub-card {
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid #252830;
            border-radius: 12px;
            padding: 14px;
            margin: 8px 0;
        }
        
        .sub-card-title {
            font-size: 0.9rem;
            font-weight: 500;
            color: #d1d5db;
            margin-bottom: 8px;
        }
        
        .sub-metric {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.3rem;
            font-weight: 600;
            color: #ffffff;
        }
        
        /* Header styles */
        .dashboard-header {
            background: linear-gradient(135deg, #1a1d24 0%, #0e1117 100%);
            border: 1px solid #2d3139;
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 24px;
        }
        
        .header-title {
            font-family: 'Outfit', sans-serif;
            font-size: 1.8rem;
            font-weight: 700;
            color: #ffffff;
            margin: 0 0 8px 0;
        }
        
        .header-subtitle {
            font-size: 1rem;
            color: #8b949e;
        }
        
        /* Navigation pills */
        .nav-pill {
            display: inline-flex;
            align-items: center;
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 0.9rem;
            color: #8b949e;
            text-decoration: none;
            transition: all 0.2s ease;
            cursor: pointer;
            margin-right: 4px;
        }
        
        .nav-pill:hover {
            background: rgba(255, 255, 255, 0.05);
            color: #ffffff;
        }
        
        .nav-pill-active {
            background: rgba(84, 160, 255, 0.15);
            color: #54a0ff;
            border: 1px solid rgba(84, 160, 255, 0.3);
        }
        
        /* Button styles */
        .action-button {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 10px 20px;
            border-radius: 10px;
            font-size: 0.9rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
            border: none;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #54a0ff 0%, #2e86de 100%);
            color: #ffffff;
        }
        
        .btn-primary:hover {
            box-shadow: 0 4px 15px rgba(84, 160, 255, 0.4);
            transform: translateY(-1px);
        }
        
        .btn-secondary {
            background: rgba(255, 255, 255, 0.05);
            color: #d1d5db;
            border: 1px solid #3d4149;
        }
        
        .btn-secondary:hover {
            background: rgba(255, 255, 255, 0.1);
        }
        
        /* Count badge */
        .count-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 24px;
            height: 24px;
            border-radius: 6px;
            background: rgba(255, 159, 67, 0.15);
            color: #ff9f43;
            font-size: 0.8rem;
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
        }
        
        /* Progress bar */
        .progress-bar-container {
            width: 100%;
            height: 6px;
            background: #1a1d24;
            border-radius: 3px;
            margin-top: 8px;
            overflow: hidden;
        }
        
        .progress-bar-fill {
            height: 100%;
            border-radius: 3px;
            transition: width 0.5s ease;
        }
        
        /* Section divider */
        .section-divider {
            height: 1px;
            background: linear-gradient(90deg, transparent 0%, #3d4149 50%, transparent 100%);
            margin: 24px 0;
        }
        
        /* Tooltip */
        .tooltip-container {
            position: relative;
            display: inline-block;
        }
        
        .tooltip-text {
            visibility: hidden;
            background: #1a1d24;
            color: #d1d5db;
            font-size: 0.8rem;
            padding: 8px 12px;
            border-radius: 8px;
            border: 1px solid #3d4149;
            position: absolute;
            z-index: 100;
            bottom: 125%;
            left: 50%;
            transform: translateX(-50%);
            white-space: nowrap;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
        }
        
        .tooltip-container:hover .tooltip-text {
            visibility: visible;
        }
        
        /* Streamlit overrides */
        .stSelectbox > div > div {
            background-color: #1a1d24;
            border-color: #3d4149;
        }
        
        .stSlider > div > div > div {
            background-color: #54a0ff;
        }
        
        div[data-testid="stExpander"] {
            background-color: #1a1d24;
            border: 1px solid #2d3139;
            border-radius: 12px;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            background-color: transparent;
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: transparent;
            border-radius: 8px;
            color: #8b949e;
            padding: 8px 16px;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: rgba(84, 160, 255, 0.15);
            color: #54a0ff;
        }
        
        /* Animation */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .animate-fade-in {
            animation: fadeIn 0.4s ease forwards;
        }
        
        /* Responsive adjustments */
        @media (max-width: 768px) {
            .risk-card {
                padding: 16px;
            }
            .metric-value {
                font-size: 1.6rem;
            }
        }
    </style>
    """, unsafe_allow_html=True)

# =============================================================================
# DATA GENERATION
# =============================================================================

@st.cache_data
def generate_sample_data(n_accounts: int = 300, seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic sample customer data for the dashboard.
    
    Args:
        n_accounts: Number of customer accounts to generate
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with customer metrics and churn indicators
    """
    np.random.seed(seed)
    random.seed(seed)
    
    regions = ["AMER", "EMEA", "APAC"]
    region_weights = [0.45, 0.35, 0.20]
    
    # Industry segments for realism
    industries = ["Technology", "Healthcare", "Finance", "Retail", "Manufacturing", "Education"]
    company_sizes = ["SMB", "Mid-Market", "Enterprise"]
    
    data = {
        "account_id": [f"ACC-{i:05d}" for i in range(1, n_accounts + 1)],
        "company_name": [f"Company {chr(65 + i % 26)}{i}" for i in range(n_accounts)],
        "region": np.random.choice(regions, n_accounts, p=region_weights),
        "industry": np.random.choice(industries, n_accounts),
        "company_size": np.random.choice(company_sizes, n_accounts, p=[0.5, 0.35, 0.15]),
        "arr_value": np.random.lognormal(mean=10.5, sigma=1.2, size=n_accounts).astype(int) * 100,
        "contract_start_date": [
            datetime.now() - timedelta(days=np.random.randint(30, 730)) 
            for _ in range(n_accounts)
        ],
        "renewal_date": [
            datetime.now() + timedelta(days=np.random.randint(-30, 365)) 
            for _ in range(n_accounts)
        ],
    }
    
    df = pd.DataFrame(data)
    
    # Generate correlated metrics (customers with poor adoption tend to have poor engagement)
    base_health = np.random.beta(5, 2, n_accounts)  # Base health score (0-1)
    
    df["core_module_adoption"] = np.clip(base_health * np.random.uniform(0.7, 1.3, n_accounts), 0.15, 1.0)
    df["onboarding_completion_pct"] = np.clip(base_health * np.random.uniform(0.8, 1.2, n_accounts), 0.2, 1.0)
    df["weekly_logins"] = np.clip(
        (base_health * 10 * np.random.uniform(0.5, 1.5, n_accounts)).astype(int), 
        0, 20
    )
    df["time_to_first_value_days"] = np.clip(
        ((1 - base_health) * 45 + np.random.normal(0, 5, n_accounts)).astype(int),
        3, 90
    )
    df["seat_utilization_pct"] = np.clip(base_health * np.random.uniform(0.6, 1.2, n_accounts), 0.1, 1.0)
    df["feature_adoption_score"] = np.clip(base_health * np.random.uniform(0.7, 1.2, n_accounts), 0.1, 1.0)
    df["support_tickets_last_quarter"] = np.clip(
        ((1 - base_health) * 8 + np.random.poisson(2, n_accounts)).astype(int),
        0, 20
    )
    df["nps_score"] = np.clip(
        (base_health * 10 + np.random.normal(0, 1.5, n_accounts)),
        0, 10
    )
    df["csm_engagement_score"] = np.clip(base_health * np.random.uniform(0.6, 1.3, n_accounts), 0.2, 1.0)
    df["training_completion_pct"] = np.clip(base_health * np.random.uniform(0.5, 1.2, n_accounts), 0.1, 1.0)
    df["api_integration_count"] = np.clip(
        (base_health * 8 * np.random.uniform(0.3, 1.5, n_accounts)).astype(int),
        0, 15
    )
    df["days_since_last_login"] = np.clip(
        ((1 - base_health) * 30 + np.random.exponential(5, n_accounts)).astype(int),
        0, 90
    )
    
    # Calculate churn probability based on metrics
    churn_score = (
        (BENCHMARKS.core_module_adoption - df["core_module_adoption"]) * 0.25 +
        (BENCHMARKS.onboarding_completion - df["onboarding_completion_pct"]) * 0.15 +
        (BENCHMARKS.weekly_logins - df["weekly_logins"]) / BENCHMARKS.weekly_logins * 0.15 +
        (df["time_to_first_value_days"] - BENCHMARKS.time_to_first_value_days) / 30 * 0.15 +
        (BENCHMARKS.seat_utilization - df["seat_utilization_pct"]) * 0.15 +
        (df["support_tickets_last_quarter"] / 10) * 0.15
    )
    
    churn_prob = 1 / (1 + np.exp(-churn_score * 3))  # Sigmoid transformation
    df["churn_probability"] = churn_prob
    df["churned"] = np.random.random(n_accounts) < churn_prob
    
    # Risk tier assignment
    df["risk_tier"] = pd.cut(
        df["churn_probability"],
        bins=[0, 0.3, 0.6, 1.0],
        labels=["Low", "Medium", "High"]
    )
    
    return df

# =============================================================================
# RISK COMPUTATION
# =============================================================================

def compute_risk_attribution(df: pd.DataFrame, benchmarks: Benchmark = BENCHMARKS) -> Dict:
    """
    Compute risk attribution scores by category and sub-metric.
    
    Args:
        df: Customer DataFrame
        benchmarks: Benchmark configuration
        
    Returns:
        Dictionary containing risk scores and attributions
    """
    # Filter to at-risk accounts (high + medium risk)
    at_risk = df[df["risk_tier"].isin(["High", "Medium"])]
    total_risk_arr = at_risk["arr_value"].sum()
    
    # Product Risk Components
    product_gaps_score = max(0, (benchmarks.core_module_adoption - df["core_module_adoption"].mean()))
    seat_util_score = max(0, (benchmarks.seat_utilization - df["seat_utilization_pct"].mean()))
    feature_adoption_score = max(0, (0.7 - df["feature_adoption_score"].mean()))
    
    product_risk = (product_gaps_score + seat_util_score + feature_adoption_score) / 3
    
    # Process Risk Components
    onboarding_score = max(0, (benchmarks.onboarding_completion - df["onboarding_completion_pct"].mean()))
    login_score = max(0, (benchmarks.weekly_logins - df["weekly_logins"].mean()) / benchmarks.weekly_logins)
    ttfv_score = max(0, (df["time_to_first_value_days"].mean() - benchmarks.time_to_first_value_days) / 30)
    
    process_risk = (onboarding_score + login_score + ttfv_score) / 3
    
    # Development/Training Risk Components
    training_score = max(0, (0.8 - df["training_completion_pct"].mean()))
    api_score = max(0, (5 - df["api_integration_count"].mean()) / 5)
    
    development_risk = (training_score + api_score) / 2
    
    # Relationship Risk Components
    nps_score = max(0, (benchmarks.nps_score - df["nps_score"].mean()) / 10)
    csm_score = max(0, (0.8 - df["csm_engagement_score"].mean()))
    support_score = min(1, df["support_tickets_last_quarter"].mean() / 10)
    
    relationship_risk = (nps_score + csm_score + support_score) / 3
    
    # Normalize to percentages (sum to 100%)
    total_raw_risk = product_risk + process_risk + development_risk + relationship_risk
    
    if total_raw_risk > 0:
        product_pct = (product_risk / total_raw_risk) * 100
        process_pct = (process_risk / total_raw_risk) * 100
        development_pct = (development_risk / total_raw_risk) * 100
        relationship_pct = (relationship_risk / total_raw_risk) * 100
    else:
        product_pct = process_pct = development_pct = relationship_pct = 25
    
    return {
        "total_risk_arr": total_risk_arr,
        "at_risk_accounts": len(at_risk),
        "high_risk_accounts": len(df[df["risk_tier"] == "High"]),
        "categories": {
            "product": {
                "percentage": round(product_pct),
                "raw_score": product_risk,
                "health_score": int(5 - min(5, product_risk * 10)),
                "sub_metrics": {
                    "product_gaps": {
                        "label": "Product Gaps",
                        "value": round(df["core_module_adoption"].mean() * 100),
                        "benchmark": int(benchmarks.core_module_adoption * 100),
                        "delta": round((df["core_module_adoption"].mean() - benchmarks.core_module_adoption) * 100),
                        "status": "below" if df["core_module_adoption"].mean() < benchmarks.core_module_adoption else "above"
                    },
                    "seat_utilization": {
                        "label": "Seat Utilization",
                        "value": round(df["seat_utilization_pct"].mean() * 100),
                        "benchmark": int(benchmarks.seat_utilization * 100),
                        "delta": round((df["seat_utilization_pct"].mean() - benchmarks.seat_utilization) * 100),
                        "status": "below" if df["seat_utilization_pct"].mean() < benchmarks.seat_utilization else "above"
                    },
                    "feature_adoption": {
                        "label": "Feature Adoption",
                        "value": round(df["feature_adoption_score"].mean() * 100),
                        "benchmark": 70,
                        "delta": round((df["feature_adoption_score"].mean() - 0.7) * 100),
                        "status": "below" if df["feature_adoption_score"].mean() < 0.7 else "above"
                    }
                }
            },
            "process": {
                "percentage": round(process_pct),
                "raw_score": process_risk,
                "health_score": int(5 - min(5, process_risk * 10)),
                "sub_metrics": {
                    "onboarding": {
                        "label": "Onboarding Completion",
                        "value": round(df["onboarding_completion_pct"].mean() * 100),
                        "benchmark": int(benchmarks.onboarding_completion * 100),
                        "delta": round((df["onboarding_completion_pct"].mean() - benchmarks.onboarding_completion) * 100),
                        "status": "below" if df["onboarding_completion_pct"].mean() < benchmarks.onboarding_completion else "above"
                    },
                    "login_frequency": {
                        "label": "Login Frequency",
                        "value": round(df["weekly_logins"].mean(), 1),
                        "benchmark": benchmarks.weekly_logins,
                        "delta": round(df["weekly_logins"].mean() - benchmarks.weekly_logins, 1),
                        "status": "below" if df["weekly_logins"].mean() < benchmarks.weekly_logins else "above"
                    },
                    "ttfv": {
                        "label": "Time to First Value",
                        "value": round(df["time_to_first_value_days"].mean()),
                        "benchmark": benchmarks.time_to_first_value_days,
                        "delta": round(df["time_to_first_value_days"].mean() - benchmarks.time_to_first_value_days),
                        "status": "above" if df["time_to_first_value_days"].mean() > benchmarks.time_to_first_value_days else "below",
                        "inverse": True  # Lower is better
                    }
                }
            },
            "development": {
                "percentage": round(development_pct),
                "raw_score": development_risk,
                "health_score": int(5 - min(5, development_risk * 10)),
                "sub_metrics": {
                    "training": {
                        "label": "Training Completion",
                        "value": round(df["training_completion_pct"].mean() * 100),
                        "benchmark": 80,
                        "delta": round((df["training_completion_pct"].mean() - 0.8) * 100),
                        "status": "below" if df["training_completion_pct"].mean() < 0.8 else "above"
                    },
                    "api_integration": {
                        "label": "API Integrations",
                        "value": round(df["api_integration_count"].mean(), 1),
                        "benchmark": 5,
                        "delta": round(df["api_integration_count"].mean() - 5, 1),
                        "status": "below" if df["api_integration_count"].mean() < 5 else "above"
                    }
                }
            },
            "relationship": {
                "percentage": round(relationship_pct),
                "raw_score": relationship_risk,
                "health_score": int(5 - min(5, relationship_risk * 10)),
                "sub_metrics": {
                    "nps": {
                        "label": "NPS Score",
                        "value": round(df["nps_score"].mean(), 1),
                        "benchmark": benchmarks.nps_score,
                        "delta": round(df["nps_score"].mean() - benchmarks.nps_score, 1),
                        "status": "below" if df["nps_score"].mean() < benchmarks.nps_score else "above"
                    },
                    "csm_engagement": {
                        "label": "CSM Engagement",
                        "value": round(df["csm_engagement_score"].mean() * 100),
                        "benchmark": 80,
                        "delta": round((df["csm_engagement_score"].mean() - 0.8) * 100),
                        "status": "below" if df["csm_engagement_score"].mean() < 0.8 else "above"
                    },
                    "support_health": {
                        "label": "Support Tickets",
                        "value": round(df["support_tickets_last_quarter"].mean(), 1),
                        "benchmark": benchmarks.support_tickets_threshold,
                        "delta": round(df["support_tickets_last_quarter"].mean() - benchmarks.support_tickets_threshold, 1),
                        "status": "above" if df["support_tickets_last_quarter"].mean() > benchmarks.support_tickets_threshold else "below",
                        "inverse": True  # Lower is better
                    }
                }
            }
        }
    }

# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_score_dots(score: int, max_score: int = 5, filled_color: str = "#ff9f43") -> str:
    """Render score as colored dots."""
    dots = []
    for i in range(max_score):
        if i < score:
            dots.append(f'<span class="score-dot score-dot-filled" style="background: {filled_color};"></span>')
        else:
            dots.append('<span class="score-dot"></span>')
    return f'<div class="score-dots">{"".join(dots)}</div>'

def render_progress_bar(value: float, max_value: float = 100, color: str = "#54a0ff") -> str:
    """Render a progress bar."""
    pct = min(100, (value / max_value) * 100)
    return f'''
    <div class="progress-bar-container">
        <div class="progress-bar-fill" style="width: {pct}%; background: {color};"></div>
    </div>
    '''

def get_risk_badge_class(percentage: int) -> str:
    """Get CSS class for risk badge based on percentage."""
    if percentage >= 35:
        return "risk-badge-high"
    elif percentage >= 25:
        return "risk-badge-medium"
    return "risk-badge-low"

def get_delta_color(delta: float, inverse: bool = False) -> str:
    """Get color for delta value."""
    if inverse:
        delta = -delta
    if delta < 0:
        return "metric-delta-negative"
    return "metric-delta-positive"

def format_currency(value: float) -> str:
    """Format value as currency."""
    if value >= 1_000_000:
        return f"${value/1_000_000:.1f}M"
    elif value >= 1_000:
        return f"${value/1_000:.0f}K"
    return f"${value:.0f}"

def render_risk_category_card(
    category_name: str,
    category_data: Dict,
    icon: str,
    expanded: bool = False
) -> None:
    """
    Render a risk category card with sub-metrics.
    
    Args:
        category_name: Display name of the category
        category_data: Dictionary containing category metrics
        icon: Emoji icon for the category
        expanded: Whether to show sub-metrics expanded
    """
    pct = category_data["percentage"]
    health = category_data["health_score"]
    badge_class = get_risk_badge_class(pct)
    
    # Determine score dot color based on health
    if health >= 4:
        dot_color = "#26de81"
    elif health >= 3:
        dot_color = "#ff9f43"
    else:
        dot_color = "#ff4d4d"
    
    # Main card HTML
    card_html = f'''
    <div class="risk-card animate-fade-in">
        <div class="risk-card-header">
            <p class="risk-card-title">{icon} {category_name}</p>
            <span class="risk-badge {badge_class}">{pct}%</span>
        </div>
        <p class="benchmark-text">Responsible for {pct}% of total risk</p>
        {render_score_dots(health, 5, dot_color)}
        <p class="metric-label" style="margin-top: 4px;">{health}/5 Health Score</p>
    </div>
    '''
    st.markdown(card_html, unsafe_allow_html=True)
    
    # Sub-metrics in expander
    with st.expander(f"View {category_name} Details", expanded=expanded):
        for metric_key, metric_data in category_data["sub_metrics"].items():
            is_inverse = metric_data.get("inverse", False)
            delta_class = get_delta_color(metric_data["delta"], is_inverse)
            delta_sign = "+" if metric_data["delta"] > 0 else ""
            
            # For inverse metrics, flip the visual indicator
            if is_inverse:
                status_text = "above benchmark ⚠️" if metric_data["status"] == "above" else "at benchmark ✓"
            else:
                status_text = "below benchmark ⚠️" if metric_data["status"] == "below" else "at benchmark ✓"
            
            sub_html = f'''
            <div class="sub-card">
                <p class="sub-card-title">{metric_data["label"]}</p>
                <p class="sub-metric">
                    {metric_data["value"]}{"%" if "pct" in metric_key or "Completion" in metric_data["label"] or "Utilization" in metric_data["label"] or "Adoption" in metric_data["label"] else ""}
                    <span class="metric-delta {delta_class}">{delta_sign}{metric_data["delta"]}</span>
                </p>
                <p class="benchmark-text">Benchmark: {metric_data["benchmark"]} • {status_text}</p>
            </div>
            '''
            st.markdown(sub_html, unsafe_allow_html=True)

def render_header(risk_data: Dict, region: str) -> None:
    """Render the dashboard header with key metrics."""
    total_risk_arr = risk_data["total_risk_arr"]
    at_risk = risk_data["at_risk_accounts"]
    high_risk = risk_data["high_risk_accounts"]
    
    header_html = f'''
    <div class="dashboard-header">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 20px;">
            <div>
                <h1 class="header-title">🎯 Root-Cause Analysis</h1>
                <p class="header-subtitle">
                    <span class="count-badge">{len(risk_data["categories"])}</span>
                    risk categories identified • 
                    <span style="color: #ff4d4d;">{high_risk} high-risk</span> accounts • 
                    Region: <strong>{region}</strong>
                </p>
            </div>
            <div style="display: flex; gap: 12px; align-items: center;">
                <span style="background: rgba(255, 159, 67, 0.15); color: #ff9f43; padding: 6px 12px; border-radius: 8px; font-size: 0.85rem; font-weight: 600;">
                    +3 new risks
                </span>
            </div>
        </div>
        <div style="margin-top: 20px; display: flex; gap: 40px; flex-wrap: wrap;">
            <div>
                <p class="metric-value" style="color: #ff4d4d;">{format_currency(total_risk_arr)}</p>
                <p class="metric-label">ARR at Risk</p>
            </div>
            <div>
                <p class="metric-value">{at_risk}</p>
                <p class="metric-label">At-Risk Accounts</p>
            </div>
            <div>
                <p class="metric-value" style="color: #ff9f43;">{high_risk}</p>
                <p class="metric-label">High Priority</p>
            </div>
        </div>
    </div>
    '''
    st.markdown(header_html, unsafe_allow_html=True)

def render_sidebar() -> Tuple[str, bool]:
    """Render the sidebar navigation and filters."""
    with st.sidebar:
        st.markdown("""
        <div style="padding: 20px 0; border-bottom: 1px solid #2d3139; margin-bottom: 20px;">
            <h2 style="font-family: 'Outfit', sans-serif; font-size: 1.4rem; font-weight: 700; color: #ffffff; margin: 0;">
                📊 ARR Risk Dashboard
            </h2>
            <p style="color: #6b7280; font-size: 0.85rem; margin-top: 4px;">Customer Success Analytics</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation
        st.markdown("### Navigation")
        nav_options = ["Overview", "Revenue Impact", "Benchmarking", "ARR Plans", "Account Details"]
        
        # Use session state for navigation
        if "nav_selection" not in st.session_state:
            st.session_state.nav_selection = "Overview"
        
        for nav in nav_options:
            is_selected = st.session_state.nav_selection == nav
            if st.button(
                f"{'◉' if is_selected else '○'} {nav}",
                key=f"nav_{nav}",
                use_container_width=True,
                type="primary" if is_selected else "secondary"
            ):
                st.session_state.nav_selection = nav
        
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        
        # Region Filter
        st.markdown("### 🌍 Region Mapping")
        region = st.selectbox(
            "Select Region",
            options=[r.value for r in Region],
            index=0,
            label_visibility="collapsed"
        )
        
        # Quick region buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("AMER", use_container_width=True):
                region = "AMER"
        with col2:
            if st.button("EMEA", use_container_width=True):
                region = "EMEA"
        with col3:
            if st.button("APAC", use_container_width=True):
                region = "APAC"
        
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        
        # Benchmark Adjustments
        st.markdown("### ⚙️ Settings")
        
        with st.expander("Adjust Benchmarks"):
            st.slider("Core Adoption %", 50, 100, 80, key="bench_adoption")
            st.slider("Onboarding %", 50, 100, 90, key="bench_onboarding")
            st.slider("Weekly Logins", 1, 10, 5, key="bench_logins")
        
        # Refresh Data
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        
        refresh = st.button("🔄 Refresh Data", use_container_width=True)
        
        # Export placeholder
        st.button("📄 Create Report", use_container_width=True, disabled=True)
        
        # Footer
        st.markdown("""
        <div style="position: fixed; bottom: 20px; left: 20px; right: 20px; max-width: 260px;">
            <p style="color: #4b5563; font-size: 0.75rem; text-align: center;">
                Last updated: Today at 2:34 PM<br>
                Data refreshes every 4 hours
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        return region, refresh

def render_risk_distribution_chart(risk_data: Dict) -> None:
    """Render a donut chart showing risk distribution."""
    categories = risk_data["categories"]
    
    labels = ["Product Risk", "Process Risk", "Development Risk", "Relationship Risk"]
    values = [
        categories["product"]["percentage"],
        categories["process"]["percentage"],
        categories["development"]["percentage"],
        categories["relationship"]["percentage"]
    ]
    colors = [COLORS["accent_red"], COLORS["accent_orange"], COLORS["accent_yellow"], COLORS["accent_purple"]]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.65,
        marker=dict(colors=colors, line=dict(color='#0e1117', width=2)),
        textinfo='percent',
        textfont=dict(size=12, color='white', family='JetBrains Mono'),
        hovertemplate="<b>%{label}</b><br>%{percent}<br>Risk Contribution<extra></extra>"
    )])
    
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(color='#8b949e', size=11)
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=20, b=60, l=20, r=20),
        height=300,
        annotations=[dict(
            text=f'<b>{sum(values)}%</b><br><span style="font-size:10px">Total Risk</span>',
            x=0.5, y=0.5,
            font=dict(size=20, color='white', family='Outfit'),
            showarrow=False
        )]
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def render_trend_chart(df: pd.DataFrame) -> None:
    """Render a trend line chart for risk over time."""
    # Generate synthetic trend data
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    
    np.random.seed(42)
    base_risk = 35
    trend = np.cumsum(np.random.randn(30) * 2) + base_risk
    trend = np.clip(trend, 10, 60)
    
    trend_df = pd.DataFrame({
        'Date': dates,
        'Risk Score': trend
    })
    
    fig = px.line(
        trend_df, x='Date', y='Risk Score',
        line_shape='spline'
    )
    
    fig.update_traces(
        line=dict(color=COLORS["accent_blue"], width=3),
        fill='tozeroy',
        fillcolor='rgba(84, 160, 255, 0.1)'
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=20, b=40, l=40, r=20),
        height=200,
        xaxis=dict(
            showgrid=False,
            color='#8b949e',
            tickfont=dict(size=10)
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.05)',
            color='#8b949e',
            tickfont=dict(size=10)
        ),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def render_top_accounts_table(df: pd.DataFrame, n: int = 5) -> None:
    """Render a table of top at-risk accounts."""
    top_risk = df.nlargest(n, 'churn_probability')[
        ['account_id', 'company_name', 'region', 'arr_value', 'churn_probability', 'risk_tier']
    ].copy()
    
    top_risk['arr_value'] = top_risk['arr_value'].apply(format_currency)
    top_risk['churn_probability'] = (top_risk['churn_probability'] * 100).round(0).astype(int).astype(str) + '%'
    top_risk.columns = ['ID', 'Company', 'Region', 'ARR', 'Risk %', 'Tier']
    
    st.markdown("""
    <style>
    .dataframe {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem !important;
    }
    .dataframe th {
        background-color: #1a1d24 !important;
        color: #8b949e !important;
        font-weight: 500 !important;
    }
    .dataframe td {
        background-color: #0e1117 !important;
        color: #e5e7eb !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.dataframe(
        top_risk,
        use_container_width=True,
        hide_index=True
    )

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""
    # Page config
    st.set_page_config(
        page_title="ARR Risk Attribution Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Inject custom CSS
    inject_custom_css()
    
    # Initialize session state
    if "data_seed" not in st.session_state:
        st.session_state.data_seed = 42
    if "widgets" not in st.session_state:
        st.session_state.widgets = []
    
    # Render sidebar and get filters
    region, refresh = render_sidebar()
    
    # Refresh data if requested
    if refresh:
        st.session_state.data_seed = random.randint(1, 10000)
        st.rerun()
    
    # Generate data
    df = generate_sample_data(n_accounts=350, seed=st.session_state.data_seed)
    
    # Filter by region
    if region != "All Regions":
        df_filtered = df[df["region"] == region].copy()
    else:
        df_filtered = df.copy()
    
    # Compute risk attribution
    risk_data = compute_risk_attribution(df_filtered)
    
    # Main content area
    render_header(risk_data, region)
    
    # Navigation tabs
    tabs = st.tabs(["📊 Overview", "📈 Revenue Impact", "🎯 Benchmarking", "📋 Account Details"])
    
    with tabs[0]:  # Overview
        # Risk category cards
        st.markdown("### Risk Attribution by Category")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            render_risk_category_card(
                "Product Risk",
                risk_data["categories"]["product"],
                "🔧"
            )
        
        with col2:
            render_risk_category_card(
                "Process Risk",
                risk_data["categories"]["process"],
                "⚙️"
            )
        
        with col3:
            render_risk_category_card(
                "Development Risk",
                risk_data["categories"]["development"],
                "📚"
            )
        
        with col4:
            render_risk_category_card(
                "Relationship Risk",
                risk_data["categories"]["relationship"],
                "🤝"
            )
        
        # Charts row
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        
        chart_col1, chart_col2 = st.columns([1, 2])
        
        with chart_col1:
            st.markdown("#### Risk Distribution")
            render_risk_distribution_chart(risk_data)
        
        with chart_col2:
            st.markdown("#### Risk Trend (30 Days)")
            render_trend_chart(df_filtered)
        
        # Top at-risk accounts
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        st.markdown("#### 🚨 Top At-Risk Accounts")
        render_top_accounts_table(df_filtered)
        
        # Add Widget button (placeholder)
        st.markdown("<br>", unsafe_allow_html=True)
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])
        with col_btn1:
            if st.button("➕ Add Widget", use_container_width=True):
                st.session_state.widgets.append(f"Widget {len(st.session_state.widgets) + 1}")
                st.toast("Widget added! (Placeholder)", icon="✅")
        with col_btn2:
            st.button("📥 Export Data", use_container_width=True, disabled=True)
    
    with tabs[1]:  # Revenue Impact
        st.markdown("### 💰 Revenue Impact Analysis")
        
        # ARR by risk tier
        arr_by_tier = df_filtered.groupby('risk_tier')['arr_value'].sum().reset_index()
        arr_by_tier.columns = ['Risk Tier', 'ARR']
        
        fig = px.bar(
            arr_by_tier, 
            x='Risk Tier', 
            y='ARR',
            color='Risk Tier',
            color_discrete_map={'Low': COLORS['accent_green'], 'Medium': COLORS['accent_orange'], 'High': COLORS['accent_red']}
        )
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(color='#8b949e'),
            yaxis=dict(color='#8b949e', gridcolor='rgba(255,255,255,0.05)'),
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Key metrics
        met_col1, met_col2, met_col3, met_col4 = st.columns(4)
        with met_col1:
            high_risk_arr = df_filtered[df_filtered['risk_tier'] == 'High']['arr_value'].sum()
            st.metric("High Risk ARR", format_currency(high_risk_arr))
        with met_col2:
            med_risk_arr = df_filtered[df_filtered['risk_tier'] == 'Medium']['arr_value'].sum()
            st.metric("Medium Risk ARR", format_currency(med_risk_arr))
        with met_col3:
            low_risk_arr = df_filtered[df_filtered['risk_tier'] == 'Low']['arr_value'].sum()
            st.metric("Low Risk ARR", format_currency(low_risk_arr))
        with met_col4:
            total_arr = df_filtered['arr_value'].sum()
            st.metric("Total ARR", format_currency(total_arr))
    
    with tabs[2]:  # Benchmarking
        st.markdown("### 🎯 Benchmark Comparison")
        
        # Radar chart for benchmarks
        categories_list = ['Core Adoption', 'Onboarding', 'Seat Utilization', 'Training', 'NPS', 'CSM Engagement']
        
        actual_values = [
            df_filtered['core_module_adoption'].mean() * 100,
            df_filtered['onboarding_completion_pct'].mean() * 100,
            df_filtered['seat_utilization_pct'].mean() * 100,
            df_filtered['training_completion_pct'].mean() * 100,
            df_filtered['nps_score'].mean() * 10,
            df_filtered['csm_engagement_score'].mean() * 100
        ]
        
        benchmark_values = [80, 90, 75, 80, 80, 80]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=actual_values,
            theta=categories_list,
            fill='toself',
            name='Actual',
            line_color=COLORS['accent_blue'],
            fillcolor='rgba(84, 160, 255, 0.2)'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=benchmark_values,
            theta=categories_list,
            fill='toself',
            name='Benchmark',
            line_color=COLORS['accent_green'],
            fillcolor='rgba(38, 222, 129, 0.1)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    gridcolor='rgba(255,255,255,0.1)'
                ),
                bgcolor='rgba(0,0,0,0)'
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5,
                font=dict(color='#8b949e')
            ),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Benchmark table
        st.markdown("#### Detailed Benchmark Comparison")
        benchmark_df = pd.DataFrame({
            'Metric': categories_list,
            'Actual': [f"{v:.1f}%" for v in actual_values],
            'Benchmark': [f"{v}%" for v in benchmark_values],
            'Gap': [f"{actual_values[i] - benchmark_values[i]:+.1f}%" for i in range(len(categories_list))]
        })
        st.dataframe(benchmark_df, use_container_width=True, hide_index=True)
    
    with tabs[3]:  # Account Details
        st.markdown("### 📋 Account Details")
        
        # Filters
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        with filter_col1:
            risk_filter = st.multiselect("Risk Tier", options=["High", "Medium", "Low"], default=["High", "Medium"])
        with filter_col2:
            industry_filter = st.multiselect("Industry", options=df_filtered['industry'].unique().tolist())
        with filter_col3:
            size_filter = st.multiselect("Company Size", options=df_filtered['company_size'].unique().tolist())
        
        # Apply filters
        display_df = df_filtered.copy()
        if risk_filter:
            display_df = display_df[display_df['risk_tier'].isin(risk_filter)]
        if industry_filter:
            display_df = display_df[display_df['industry'].isin(industry_filter)]
        if size_filter:
            display_df = display_df[display_df['company_size'].isin(size_filter)]
        
        # Display filtered data
        st.markdown(f"**Showing {len(display_df)} accounts**")
        
        display_cols = ['account_id', 'company_name', 'region', 'industry', 'company_size', 
                       'arr_value', 'core_module_adoption', 'onboarding_completion_pct', 
                       'churn_probability', 'risk_tier']
        
        display_df_formatted = display_df[display_cols].copy()
        display_df_formatted['arr_value'] = display_df_formatted['arr_value'].apply(format_currency)
        display_df_formatted['core_module_adoption'] = (display_df_formatted['core_module_adoption'] * 100).round(0).astype(int).astype(str) + '%'
        display_df_formatted['onboarding_completion_pct'] = (display_df_formatted['onboarding_completion_pct'] * 100).round(0).astype(int).astype(str) + '%'
        display_df_formatted['churn_probability'] = (display_df_formatted['churn_probability'] * 100).round(0).astype(int).astype(str) + '%'
        
        display_df_formatted.columns = ['ID', 'Company', 'Region', 'Industry', 'Size', 'ARR', 
                                        'Adoption', 'Onboarding', 'Risk %', 'Tier']
        
        st.dataframe(
            display_df_formatted,
            use_container_width=True,
            hide_index=True,
            height=400
        )
        
        # Export button
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="at_risk_accounts.csv",
            mime="text/csv"
        )
    
    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 40px 0 20px 0; border-top: 1px solid #2d3139; margin-top: 40px;">
        <p style="color: #4b5563; font-size: 0.85rem;">
            ARR Risk Attribution Dashboard • Built with Streamlit<br>
            <span style="font-size: 0.75rem;">For demonstration purposes. Replace with real data integration.</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()

# =============================================================================
# FUTURE ENHANCEMENTS (TODO)
# =============================================================================
"""
ML Integration Points:
----------------------
1. Replace rule-based risk scoring with scikit-learn model:
   - from sklearn.ensemble import GradientBoostingClassifier
   - Train on historical churn data
   - Use SHAP values for explainable attribution

2. Add time-series forecasting for ARR risk:
   - from prophet import Prophet
   - Predict risk trends 30/60/90 days out

3. Clustering for account segmentation:
   - from sklearn.cluster import KMeans
   - Identify risk archetypes

4. Anomaly detection for early warning:
   - from sklearn.ensemble import IsolationForest
   - Flag accounts with unusual behavior patterns

Database Integration:
--------------------
- Connect to real customer data via SQLAlchemy
- Implement caching with Redis
- Add real-time updates via WebSockets

Additional Features:
-------------------
- PDF report generation with ReportLab
- Email alerts for risk threshold breaches
- Slack/Teams integration for notifications
- Role-based access control
"""
