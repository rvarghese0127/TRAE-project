# ARR Risk Attribution Dashboard

A production-grade Streamlit dashboard for customer churn risk attribution and root-cause analysis. Designed with a dark theme, modern SaaS aesthetics, and interactive data exploration.

![Dashboard Preview](dashboard_preview.png)

## Features

### 🎯 Risk Attribution
- **Product Risk**: Core module adoption, seat utilization, feature adoption gaps
- **Process Risk**: Onboarding completion, login frequency, time to first value
- **Development Risk**: Training completion, API integration depth
- **Relationship Risk**: NPS scores, CSM engagement, support ticket health

### 📊 Interactive Visualizations
- Risk distribution donut chart
- 30-day risk trend line chart
- Benchmark radar comparison
- ARR by risk tier bar charts

### 🔧 Filtering & Controls
- Region filtering (AMER, EMEA, APAC)
- Risk tier filtering
- Industry and company size filters
- Adjustable benchmarks via sliders

### 📋 Account Management
- Top at-risk accounts table
- Detailed account explorer with export
- CSV download functionality

## Installation

### Prerequisites
- Python 3.10+
- pip or conda

### Quick Start

1. **Clone or download the repository**

2. **Install dependencies**
```bash
pip install streamlit pandas numpy plotly
```

3. **Run the dashboard**
```bash
streamlit run app.py
```

4. **Open in browser**
Navigate to `http://localhost:8501`

## Configuration

### Streamlit Theme
The dark theme is configured in `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#54a0ff"
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#1a1d24"
textColor = "#ffffff"
font = "sans serif"
```

### Benchmark Adjustment
Default benchmarks can be modified in the sidebar or in the code:

```python
@dataclass
class Benchmark:
    core_module_adoption: float = 0.80
    onboarding_completion: float = 0.90
    weekly_logins: int = 5
    time_to_first_value_days: int = 14
    seat_utilization: float = 0.75
    support_tickets_threshold: int = 3
    nps_score: float = 8.0
```

## Data Generation

The dashboard uses synthetic data for demonstration. To connect to real data:

1. Replace `generate_sample_data()` with your data source
2. Ensure your DataFrame has the required columns:
   - `account_id`, `company_name`, `region`
   - `arr_value`, `core_module_adoption`
   - `onboarding_completion_pct`, `weekly_logins`
   - `time_to_first_value_days`, `seat_utilization_pct`
   - `support_tickets_last_quarter`, `nps_score`
   - `csm_engagement_score`, `training_completion_pct`

## Future Enhancements

### ML Integration Points
- Replace rule-based scoring with GradientBoostingClassifier
- Add SHAP values for explainable risk attribution
- Implement Prophet for ARR risk forecasting
- Use IsolationForest for anomaly detection

### Production Features
- Database integration (PostgreSQL, Snowflake)
- Redis caching for performance
- PDF report generation
- Slack/Teams notifications
- Role-based access control

## Project Structure

```
.
├── app.py                  # Main Streamlit application
├── .streamlit/
│   └── config.toml         # Streamlit configuration
├── README.md               # This file
└── requirements.txt        # Python dependencies
```

## Requirements

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.17.0
```

## License

MIT License - Feel free to use and modify for your own projects.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

Built with ❤️ using Streamlit
