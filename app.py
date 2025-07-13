import streamlit as st
import pickle
import pandas as pd
from nav import home, predict, result_visualization, logs, data_analysis

# Load the precomputed models, metrics, and label encoder
models_file = 'precomputation/models.pkl'
metrics_file = 'precomputation/metrics.pkl'
le_file = 'precomputation/label_encoder.pkl'

with open(models_file, 'rb') as f:
    models = pickle.load(f)

with open(metrics_file, 'rb') as f:
    metrics = pickle.load(f)

with open(le_file, 'rb') as f:
    le = pickle.load(f)

# Load the dataset for feature names
dataset_path = 'dataset/crop_recommendation.csv'
df = pd.read_csv(dataset_path)
X = df.drop('label', axis=1)

# Custom CSS for better navigation bar with animations and icons
st.markdown("""
    <style>
    .navbar {
        display: flex;
        justify-content: space-around;
        background-color: #f8f9fa;
        padding: 10px;
        border-bottom: 2px solid #dee2e6;
        animation: fadeInDown 1s ease-out;
    }
    .navbar a {
        text-decoration: none;
        color: #333;
        font-weight: bold;
        display: flex;
        align-items: center;
    }
    .navbar a:hover {
        color: #007bff;
    }
    .navbar i {
        margin-right: 5px;
    }
    @keyframes fadeInDown {
        0% {
            opacity: 0;
            transform: translateY(-10px);
        }
        100% {
            opacity: 1;
            transform: translateY(0);
        }
    }
    </style>
""", unsafe_allow_html=True)

# Create navigation bar with links targeted to _self (same window) and icons
st.markdown("""
    <div class="navbar">
        <a href="?page=home" id="link-home" target="_self"><i class="fa fa-home"></i> Home</a>
        <a href="?page=predict" id="link-predict" target="_self"><i class="fa fa-pie-chart"></i> Predict</a>
        <a href="?page=resultVisualization" id="link-result-visualization" target="_self"><i class="fa fa-bar-chart"></i> Result Visualization</a>
        <a href="?page=dataAnalysis" id="link-data-analysis" target="_self"><i class="fa fa-line-chart"></i> Data Analysis</a>
        <a href="?page=logs" id="link-logs" target="_self"><i class="fa fa-file-text"></i> Logs</a>
    </div>
""", unsafe_allow_html=True)

# Determine the selected page based on the query parameters
query_params = st.query_params
page = query_params.get("page", "home")

if page == "home":
    home.show()
elif page == "predict":
    predict.show(models, metrics, le, X.columns)
elif page == "resultVisualization":
    result_visualization.show(models, metrics, le, X)
elif page == "dataAnalysis":
    data_analysis.main()
elif page == "logs":
    logs.show_logs()
else:
    home.show()
