import time
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

import json
from datetime import datetime
import warnings
import sys
import os
import io
import uuid
from typing import Dict, List, Optional

warnings.filterwarnings('ignore')

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index

plt.ioff()
matplotlib_backend = plt.get_backend()
if 'Qt' in matplotlib_backend or 'TkAgg' in matplotlib_backend:
    plt.switch_backend('Agg')

# Set page configuration
st.set_page_config(
    page_title="Lung Cancer Survival Analysis System",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styles
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #667eea;
        --secondary-color: #764ba2;
        --success-color: #10ac84;
        --warning-color: #ff9f43;
        --danger-color: #ee5a52;
        --info-color: #3742fa;
        --light-bg: #f8f9fa;
        --card-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container styling */
    .main-container {
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: var(--card-shadow);
    }
    
    /* Step indicators */
    .step-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 2rem 0;
        padding: 1rem;
        background: white;
        border-radius: 10px;
        box-shadow: var(--card-shadow);
    }
    
    .step-item {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin: 0 1rem;
        padding: 1rem;
        border-radius: 10px;
        min-width: 120px;
        transition: all 0.3s ease;
    }
    
    .step-item.active {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white;
        transform: scale(1.05);
    }
    
    .step-item.completed {
        background: linear-gradient(135deg, var(--success-color), #06a077);
        color: white;
    }
    
    .step-item.inactive {
        background: #e9ecef;
        color: #6c757d;
    }
    
    .step-number {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
        background: rgba(255,255,255,0.2);
    }
    
    .step-title {
        font-weight: bold;
        font-size: 0.9rem;
        text-align: center;
    }
    
    /* Enhanced cards */
    .analysis-card {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: var(--card-shadow);
        border-left: 5px solid var(--primary-color);
        transition: transform 0.3s ease;
    }
    
    .analysis-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .patient-card {
        background: linear-gradient(145deg, #ffffff, #f8f9fa);
        border: none;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: var(--card-shadow);
        transition: all 0.3s ease;
    }
    
    .patient-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    }
    
    /* Risk indicators */
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 25px;
        text-align: center;
        font-weight: bold;
        font-size: 0.9rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 8px rgba(238, 90, 82, 0.3);
    }
    
    .risk-low {
        background: linear-gradient(135deg, #51cf66, #40c057);
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 25px;
        text-align: center;
        font-weight: bold;
        font-size: 0.9rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 8px rgba(64, 192, 87, 0.3);
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: var(--card-shadow);
        text-align: center;
        margin: 1rem;
        border-top: 4px solid var(--primary-color);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        border-top-color: var(--secondary-color);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: var(--primary-color);
        margin: 0.5rem 0;
    }
    
    .metric-label {
        color: #6c757d;
        font-size: 1rem;
        font-weight: 500;
    }
    
    /* Sidebar styling */
    .sidebar-title {
        color: var(--primary-color);
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 1.5rem;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
        border-radius: 10px;
    }
    
    /* Alert boxes */
    .info-box {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        border-left: 5px solid var(--info-color);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: #1565c0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #e8f5e8, #c8e6c9);
        border-left: 5px solid var(--success-color);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: #2e7d32;
    }
    
    /* Data upload area */
    .upload-area {
        border: 2px dashed var(--primary-color);
        border-radius: 15px;
        padding: 3rem;
        text-align: center;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.05), rgba(118, 75, 162, 0.05));
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: var(--secondary-color);
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
    }
    
    /* Patient selection enhancements */
    .patient-selection-box {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: var(--card-shadow);
        border-left: 4px solid var(--info-color);
    }
    
    .quick-select-buttons {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.05), rgba(118, 75, 162, 0.05));
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .selection-status {
        background: linear-gradient(135deg, #e8f5e8, #c8e6c9);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid var(--success-color);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff8e1, #ffecb3);
        border-left: 5px solid var(--warning-color);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: #f57c00;
    }
    
    /* Navigation buttons */
    .nav-button {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
    }
    
    .nav-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Data upload area */
    .upload-area {
        border: 2px dashed var(--primary-color);
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.05), rgba(118, 75, 162, 0.05));
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: var(--secondary-color);
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
    }
    
    /* Analysis selection */
    .analysis-option {
        background: white;
        border: 2px solid #e9ecef;
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem;
        cursor: pointer;
        transition: all 0.3s ease;
        text-align: center;
    }
    
    .analysis-option:hover {
        border-color: var(--primary-color);
        transform: translateY(-5px);
        box-shadow: var(--card-shadow);
    }
    
    .analysis-option.selected {
        border-color: var(--primary-color);
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
    }
    
    /* Progress bar */
    .progress-container {
        background: #e9ecef;
        border-radius: 10px;
        height: 8px;
        margin: 1rem 0;
        overflow: hidden;
    }
    
    .progress-bar {
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        height: 100%;
        border-radius: 10px;
        transition: width 0.3s ease;
    }
    
    /* Table styling */
    .stTable {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: var(--card-shadow);
    }
</style>
""", unsafe_allow_html=True)

class SimpleHRPFSModelPackage:
    def __init__(self):
        self.model_name = "HRPFS_Simple_Package"
        self.model_version = "1.0"
        self.creation_date = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

        self.best_params = None
        self.feature_names = None
        self.scaler_params = None
        self.roc_threshold = None

        self.performance_metrics = None
        self.roc_metrics = None

        self.training_stats = None

    @classmethod
    def load_package(cls, filepath):
        st.write(f"Loading simple HRPFS model package from {filepath}...")

        with open(filepath, 'rb') as f:
            model_package = pickle.load(f)

        print(f"Simple model package loaded successfully!")
        print(f"Model info: {model_package.model_name} v{model_package.model_version}")
        print(f"Creation date: {model_package.creation_date}")
        print(f"Model parameters: {model_package.best_params}")

        return model_package


class HRPFSModelReconstructor:
    def __init__(self, simple_package):
        self.package = simple_package
        self.cox_model = None
        self.scaler = None
        self._rebuild_components()
        
    def _rebuild_components(self):
        if self.package.scaler_params is None:
            raise ValueError("Missing scaler_params in simple model package. Please ensure package is correctly created.")
        if self.package.best_params is None:
            raise ValueError("Missing best_params in simple model package. Please ensure package is correctly created.")
        if self.package.feature_names is None:
            raise ValueError("Missing feature_names in simple model package. Please ensure package is correctly created.")
            
        self.scaler = StandardScaler()
        self.scaler.mean_ = np.array(self.package.scaler_params['mean'])
        self.scaler.scale_ = np.array(self.package.scaler_params['scale'])
        self.scaler.n_features_in_ = len(self.package.feature_names)
        
        self.cox_model = CoxPHFitter(
            penalizer=self.package.best_params['penalizer'],
            l1_ratio=self.package.best_params['l1_ratio']
        )


    def fit_cox_model(self, X_train, y_train_time, y_train_event):
        X_train_scaled = self.scaler.transform(X_train)
        train_df = pd.DataFrame(X_train_scaled, columns=self.package.feature_names)
        train_df['Status'] = y_train_event.astype(int)
        train_df['Time'] = y_train_time
        self.cox_model.fit(train_df, duration_col='Time', event_col='Status')
        st.write(f"C-index: {self.cox_model.concordance_index_:.4f}")


    def calculate_risk_score(self, X, return_percentile=True):
        if not isinstance(X, pd.DataFrame):
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
            X = pd.DataFrame(X, columns=self.package.feature_names)
        
        X_scaled = self.scaler.transform(X[self.package.feature_names])
        pred_df = pd.DataFrame(X_scaled, columns=self.package.feature_names)
        pred_df['Status'] = 1
        pred_df['Time'] = 365
        
        risk_scores = self.cox_model.predict_partial_hazard(pred_df).values
        risk_groups = np.where(risk_scores > self.package.roc_threshold, 'High Risk', 'Low Risk')
        
        results = {'risk_scores': risk_scores, 'risk_groups': risk_groups}
        
        if return_percentile:
            risk_percentiles = np.array([np.sum(risk_scores <= score) / len(risk_scores) * 100 
                                       for score in risk_scores])
            results['risk_percentiles'] = risk_percentiles
            
        return results
    
    def predict_survival_function(self, X, time_points=None):
        if not isinstance(X, pd.DataFrame):
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
            X = pd.DataFrame(X, columns=self.package.feature_names)
        
        X_scaled = self.scaler.transform(X[self.package.feature_names])
        pred_df = pd.DataFrame(X_scaled, columns=self.package.feature_names)
        pred_df['Status'] = 1
        pred_df['Time'] = 365
        
        survival_functions = self.cox_model.predict_survival_function(pred_df)
        
        if time_points is not None:
            survival_at_times = []
            for time_point in time_points:
                survival_probs = []
                for j in range(len(X)):
                    sf = survival_functions[j]
                    closest_time = min(sf.index, key=lambda x: abs(x - time_point))
                    survival_probs.append(sf.loc[closest_time])
                survival_at_times.append(survival_probs)
            return np.array(survival_at_times).T
        
        return survival_functions
    



class SimpleHROSModelPackage:
    def __init__(self):
        self.model_name = "HROS_Simple_Package"
        self.model_version = "1.0"
        self.creation_date = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

        self.best_params = None
        self.feature_names = None
        self.scaler_params = None
        self.roc_threshold = None

        self.performance_metrics = None
        self.roc_metrics = None

        self.training_stats = None

    @classmethod
    def load_package(cls, filepath):
        print(f"Loading simple HROS model package from {filepath}...")

        with open(filepath, 'rb') as f:
            model_package = pickle.load(f)

        print(f"Simple model package loaded successfully!")
        print(f"Model info: {model_package.model_name} v{model_package.model_version}")
        print(f"Creation date: {model_package.creation_date}")
        print(f"Model parameters: {model_package.best_params}")

        return model_package


class HROSModelReconstructor:
    def __init__(self, simple_package):
        self.package = simple_package
        self.cox_model = None
        self.scaler = None
        self._rebuild_components()

    def _rebuild_components(self):
        if self.package.scaler_params is None:
            raise ValueError(
                "Missing scaler_params in simple model package. Please ensure package is correctly created.")
        if self.package.best_params is None:
            raise ValueError("Missing best_params in simple model package. Please ensure package is correctly created.")
        if self.package.feature_names is None:
            raise ValueError(
                "Missing feature_names in simple model package. Please ensure package is correctly created.")

        self.scaler = StandardScaler()
        self.scaler.mean_ = np.array(self.package.scaler_params['mean'])
        self.scaler.scale_ = np.array(self.package.scaler_params['scale'])
        self.scaler.n_features_in_ = len(self.package.feature_names)

        self.cox_model = CoxPHFitter(
            penalizer=self.package.best_params['penalizer'],
            l1_ratio=self.package.best_params['l1_ratio']
        )

    def fit_cox_model(self, X_train, y_train_time, y_train_event):
        X_train_scaled = self.scaler.transform(X_train)
        train_df = pd.DataFrame(X_train_scaled, columns=self.package.feature_names)
        train_df['Status'] = y_train_event.astype(int)
        train_df['Time'] = y_train_time
        self.cox_model.fit(train_df, duration_col='Time', event_col='Status')
        print(f"Cox model reconstruction completed! C-index: {self.cox_model.concordance_index_:.4f}")

    def calculate_risk_score(self, X, return_percentile=True):
        if not isinstance(X, pd.DataFrame):
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
            X = pd.DataFrame(X, columns=self.package.feature_names)

        X_scaled = self.scaler.transform(X[self.package.feature_names])
        pred_df = pd.DataFrame(X_scaled, columns=self.package.feature_names)
        pred_df['Status'] = 1
        pred_df['Time'] = 365

        risk_scores = self.cox_model.predict_partial_hazard(pred_df).values
        risk_groups = np.where(risk_scores > self.package.roc_threshold, 'High Risk', 'Low Risk')

        results = {'risk_scores': risk_scores, 'risk_groups': risk_groups}

        if return_percentile:
            risk_percentiles = np.array([np.sum(risk_scores <= score) / len(risk_scores) * 100
                                         for score in risk_scores])
            results['risk_percentiles'] = risk_percentiles

        return results

    def predict_survival_function(self, X, time_points=None):
        if not isinstance(X, pd.DataFrame):
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
            X = pd.DataFrame(X, columns=self.package.feature_names)

        X_scaled = self.scaler.transform(X[self.package.feature_names])
        pred_df = pd.DataFrame(X_scaled, columns=self.package.feature_names)
        pred_df['Status'] = 1
        pred_df['Time'] = 365

        survival_functions = self.cox_model.predict_survival_function(pred_df)

        if time_points is not None:
            survival_at_times = []
            for time_point in time_points:
                survival_probs = []
                for j in range(len(X)):
                    sf = survival_functions[j]
                    closest_time = min(sf.index, key=lambda x: abs(x - time_point))
                    survival_probs.append(sf.loc[closest_time])
                survival_at_times.append(survival_probs)
            return np.array(survival_at_times).T

        return survival_functions




def load_model(model_path):
    st.write(f"Detecting model package type: {model_path}")

    try:
        with open(model_path, 'rb') as f:
            model_package = pickle.load(f)

        if hasattr(model_package, 'cox_model') and model_package.cox_model is not None:
            st.write("Full model package detected, using directly")
            return model_package

        elif hasattr(model_package, 'best_params') and hasattr(model_package, 'feature_names'):
            print("Simple model package detected, preparing reconstruction")
            if (model_package.best_params is None or
                model_package.feature_names is None or
                model_package.scaler_params is None):
                raise ValueError("Simple model package parameters incomplete, possibly empty or corrupted package")
            # 检查模型类型来选择正确的重构器
            if "HRPFS" in model_path or "PFS" in model_path:
                return HRPFSModelReconstructor(model_package)
            else:
                return HROSModelReconstructor(model_package)

        else:
            print("Unknown model package format, trying to use as full package")
            return model_package

    except Exception as e:
        print(f"Failed to load model package: {e}")
        if "numpy._core" in str(e):
            print("Numpy version issue detected, trying compatibility handling...")
            try:
                import numpy as np
                import sys
                # Add compatibility handling
                return model_package
            except ImportError:
                print("Unable to resolve numpy compatibility issues")
                raise e
        else:
            raise e


def plot_survival_curves(survival_probs, time_points, patient_ids):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Use a beautiful color palette
    colors = plt.cm.tab20(np.linspace(0, 1, len(patient_ids)))
    
    # Plot each patient's survival curve
    for i, (patient_id, color) in enumerate(zip(patient_ids, colors)):
        ax.plot(time_points, survival_probs[i], 
                label=f'Patient {patient_id+1}', linewidth=3, color=color,
                marker='o', markersize=4, markerfacecolor='white', 
                markeredgecolor=color, markeredgewidth=1.5, alpha=0.8)
    
    # Customize the plot
    ax.set_xlabel('Time (months)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Progression-Free Survival Probability', fontsize=14, fontweight='bold')
    ax.set_title('Personalized Progression-Free Survival Curves\n(HRPFS Model Predictions)', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Beautiful legend
    legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                      frameon=True, fancybox=True, shadow=True,
                      fontsize=10, title='Patients', title_fontsize=12)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    # Grid and styling
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, max(time_points))
    ax.set_ylim(0, 1.02)
    
    # Add percentage labels on y-axis
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # Add subtle background color
    ax.set_facecolor('#fafafa')
    
    plt.tight_layout()
    return fig


def plot_risk_scores(risk_scores, risk_groups, patient_ids):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Beautiful color scheme
    colors = ['#ff4757' if group == 'High Risk' else '#2ed573' for group in risk_groups]
    
    # Create bar plot with gradient effect
    bars = ax.bar(range(len(patient_ids)), risk_scores, color=colors, alpha=0.8, 
                  edgecolor='white', linewidth=2, width=0.8)
    
    # Add value labels on bars
    for i, (bar, score, group) in enumerate(zip(bars, risk_scores, risk_groups)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=10)
        
        # Add risk group label below bar
        ax.text(bar.get_x() + bar.get_width()/2., -0.05,
                group, ha='center', va='top', 
                fontsize=9, style='italic')
    
    # Customize the plot
    ax.set_xlabel('Patient ID', fontsize=14, fontweight='bold')
    ax.set_ylabel('Risk Score', fontsize=14, fontweight='bold')
    ax.set_title('Progression-Free Survival Risk Assessment\n(HRPFS Model Predictions)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(range(len(patient_ids)))
    ax.set_xticklabels([f'Patient {pid+1}' for pid in patient_ids], 
                       fontsize=12, fontweight='bold')
    
    # Add median line
    median_risk = np.median(risk_scores)
    ax.axhline(y=median_risk, color='#3742fa', linestyle='--', linewidth=2, alpha=0.8, 
               label=f'Median Risk: {median_risk:.3f}')
    
    # Beautiful legend
    high_risk_patch = plt.Rectangle((0, 0), 1, 1, facecolor='#ff4757', alpha=0.8, label='High Risk')
    low_risk_patch = plt.Rectangle((0, 0), 1, 1, facecolor='#2ed573', alpha=0.8, label='Low Risk')
    median_patch = plt.Line2D([0], [0], color='#3742fa', linestyle='--', linewidth=2, 
                             label=f'Median: {median_risk:.3f}')
    
    legend = ax.legend(handles=[high_risk_patch, low_risk_patch, median_patch], 
                      loc='upper right', frameon=True, fancybox=True, shadow=True,
                      fontsize=11, title='Risk Categories', title_fontsize=12)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    # Grid and styling
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_facecolor('#fafafa')
    
    # Set y-axis limits with some padding
    y_max = max(risk_scores) * 1.2
    ax.set_ylim(0, y_max)
    
    plt.tight_layout()
    return fig


def generate_individual_survival_curve(survival_probs, time_points, patient_id):
    """生成并显示生存曲线"""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    # 绘制曲线（保持原有代码不变）
    ax.plot(time_points, survival_probs,
            linewidth=4, color='#2E86AB', marker='o', markersize=8,
            markerfacecolor='white', markeredgecolor='#2E86AB',
            markeredgewidth=2, alpha=0.9, label=f'Patient {patient_id}')

    # 添加关键时间点标注（保持原有代码不变）
    key_times = [12, 24, 36, 60]
    for t in key_times:
        if t <= max(time_points):
            idx = time_points.index(t) if t in time_points else min(range(len(time_points)),
                                                                    key=lambda i: abs(time_points[i] - t))
            prob = survival_probs[idx]
            ax.annotate(f'{t}m: {prob:.2f}',
                        xy=(t, prob), xytext=(t, prob + 0.1),
                        arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                        fontsize=10, fontweight='bold', ha='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    # 设置图表属性（保持原有代码不变）
    ax.set_xlabel('Time (months)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Progression-Free Survival Probability', fontsize=14, fontweight='bold')
    ax.set_title(f'Patient {patient_id} Personalized Progression-Free Survival Curve\n(HRPFS Model)',
                 fontsize=16, fontweight='bold', pad=20)

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, max(time_points))
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.set_facecolor('#fafafa')
    ax.legend(fontsize=12, loc='lower left')

    plt.tight_layout()
    return fig


def generate_individual_risk_plot(risk_score, risk_group, patient_id, risk_percentile):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    
    color = '#ff4757' if risk_group == 'High Risk' else '#2ed573'
    bar = ax1.bar([f'Patient {patient_id}'], [risk_score], color=color, alpha=0.8, 
                  edgecolor='white', linewidth=3, width=0.6)
    
    ax1.text(0, risk_score + 0.05, f'{risk_score:.4f}', ha='center', va='bottom', 
             fontweight='bold', fontsize=14)
    ax1.text(0, -0.1, risk_group, ha='center', va='top', 
             fontsize=12, style='italic', fontweight='bold')
    
    ax1.set_ylabel('Risk Score', fontsize=14, fontweight='bold')
    ax1.set_title(f'Patient {patient_id} Risk Score\n({risk_group})', 
                 fontsize=14, fontweight='bold', pad=20)
    ax1.set_ylim(0, risk_score * 1.3)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_facecolor('#fafafa')
    
    percentile = risk_percentile / 100
    remaining = 1 - percentile
    
    colors = ['#ff6b6b', '#95e1d3']
    labels = [f'Higher than {risk_percentile:.1f}% patients', f'Other patients']
    sizes = [percentile, remaining]
    
    wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                      startangle=90, textprops={'fontsize': 11})
    
    ax2.set_title(f'Patient {patient_id} Risk Percentile\n(Position among all patients)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig


# Patient management related functions
def save_patient_archive(archive_data: Dict, filename: str):
    """Save patient archive to file"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(archive_data, f, ensure_ascii=False, indent=2, default=str)
        return True
    except Exception as e:
        st.error(f"Failed to save archive: {str(e)}")
        return False

def load_patient_archive(filename: str) -> Optional[Dict]:
    """Load patient archive from file"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load archive: {str(e)}")
        return None

def get_archive_files() -> List[str]:
    """Get list of archive files in current directory"""
    return [f for f in os.listdir('.') if f.endswith('_patient_archive.json')]

def format_risk_display(risk_group: str) -> str:
    """Format risk display"""
    if risk_group == "High Risk":
        return '<div class="risk-high">🔴 High Risk</div>'
    else:
        return '<div class="risk-low">🟢 Low Risk</div>'

def create_patient_summary_card(patient_data: Dict) -> str:
    """Create patient summary card HTML"""
    risk_display = format_risk_display(patient_data.get('risk_group', 'Unknown'))
    
    return f"""
    <div class="patient-card">
        <h5>👤 {patient_data.get('name', 'Unknown Patient')}</h5>
        <p style="margin: 0.2rem 0;"><strong>Analysis Type:</strong> {patient_data.get('analysis_type', 'Unknown')}</p>
        <p style="margin: 0.2rem 0;"><strong>Risk Score:</strong> {patient_data.get('risk_score', 'N/A')}</p>
        {risk_display}
        <p style="margin: 0.2rem 0;"><strong>Risk Percentile:</strong> {patient_data.get('risk_percentile', 'N/A')}%</p>
        <p style="margin: 0.2rem 0;"><strong>Analysis Time:</strong> {patient_data.get('analysis_time', 'Unknown')}</p>
    </div>
    """

def patient_management_interface():
    """Personalized patient management interface"""
    st.markdown('<div class="main-header"><h1>🏥 Personalized Patient Management System</h1></div>', unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📋 Patient List", "💾 Archive Management", "📊 Statistics Overview"])
    
    with tab1:
        st.markdown("### Currently Managed Patients")
        
        if 'patient_management' not in st.session_state:
            st.session_state['patient_management'] = {}
        
        if st.session_state['patient_management']:
            # Display patient cards
            for patient_id, patient_data in st.session_state['patient_management'].items():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(create_patient_summary_card(patient_data), unsafe_allow_html=True)
                
                with col2:
                    st.markdown("**Actions**")
                    if st.button(f"Delete", key=f"delete_{patient_id}"):
                        del st.session_state['patient_management'][patient_id]
                        st.rerun()
                    
                    if st.button(f"Details", key=f"detail_{patient_id}"):
                        st.session_state[f'show_detail_{patient_id}'] = True
                
                # Show detailed information
                if st.session_state.get(f'show_detail_{patient_id}', False):
                    with st.expander(f"Patient {patient_data.get('name', 'Unknown')} Details", expanded=True):
                        st.json(patient_data)
                        if st.button("Close Details", key=f"close_{patient_id}"):
                            st.session_state[f'show_detail_{patient_id}'] = False
                            st.rerun()
        else:
            st.info("No patients currently managed. Please analyze patients and add them to the management list first.")
    
    with tab2:
        st.markdown("### Archive Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 💾 Save Archive")
            archive_name = st.text_input("Archive Name", placeholder="Enter archive name")
            
            if st.button("Save Current Patient List"):
                if archive_name:
                    filename = f"{archive_name}_patient_archive.json"
                    archive_data = {
                        'name': archive_name,
                        'created_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'patients': st.session_state.get('patient_management', {}),
                        'total_patients': len(st.session_state.get('patient_management', {}))
                    }
                    
                    if save_patient_archive(archive_data, filename):
                        st.success(f"✅ Archive '{archive_name}' saved successfully!")
                        st.markdown(f'<div class="archive-info">📁 Archive saved as: {filename}</div>', unsafe_allow_html=True)
                else:
                    st.warning("Please enter archive name")
        
        with col2:
            st.markdown("#### 📂 Load Archive")
            archive_files = get_archive_files()
            
            if archive_files:
                selected_archive = st.selectbox("Select Archive File", archive_files)
                
                if st.button("Load Selected Archive"):
                    archive_data = load_patient_archive(selected_archive)
                    if archive_data:
                        st.session_state['patient_management'] = archive_data.get('patients', {})
                        st.success(f"✅ Archive '{archive_data.get('name', 'Unknown')}' loaded successfully!")
                        st.markdown(f'<div class="archive-info">📊 Loaded {archive_data.get("total_patients", 0)} patient records</div>', unsafe_allow_html=True)
                        st.rerun()
            else:
                st.info("No archive files available")
        
        # Show existing archive information
        if archive_files:
            st.markdown("#### 📋 Existing Archives")
            for archive_file in archive_files:
                archive_data = load_patient_archive(archive_file)
                if archive_data:
                    with st.expander(f"📁 {archive_data.get('name', 'Unknown Archive')}"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Patient Count", archive_data.get('total_patients', 0))
                        with col2:
                            st.metric("Created Time", archive_data.get('created_time', 'Unknown'))
                        with col3:
                            if st.button("Delete Archive", key=f"del_archive_{archive_file}"):
                                try:
                                    os.remove(archive_file)
                                    st.success("Archive deleted")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Delete failed: {str(e)}")
    
    with tab3:
        st.markdown("### Statistics Overview")
        
        if st.session_state.get('patient_management'):
            patients = st.session_state['patient_management']
            
            # Overall statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Total Patients", len(patients))
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Risk group statistics
            risk_counts = {}
            analysis_types = {}
            
            for patient_data in patients.values():
                risk_group = patient_data.get('risk_group', 'Unknown')
                analysis_type = patient_data.get('analysis_type', 'Unknown')
                
                risk_counts[risk_group] = risk_counts.get(risk_group, 0) + 1
                analysis_types[analysis_type] = analysis_types.get(analysis_type, 0) + 1
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("High Risk Patients", risk_counts.get('High Risk', 0))
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Low Risk Patients", risk_counts.get('Low Risk', 0))
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Archive Files", len(get_archive_files()))
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Risk distribution charts
            if risk_counts:
                st.markdown("#### Risk Distribution")
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Risk group pie chart - 确保颜色对应正确
                risk_labels = list(risk_counts.keys())
                risk_values = list(risk_counts.values())
                risk_colors = []
                
                # 根据风险等级分配颜色：High Risk=红色，Low Risk=绿色
                for label in risk_labels:
                    if 'High Risk' in label:
                        risk_colors.append('#ff4757')  # 红色
                    else:
                        risk_colors.append('#2ed573')  # 绿色
                
                ax1.pie(risk_values, labels=risk_labels, autopct='%1.1f%%', colors=risk_colors)
                ax1.set_title('Risk Group Distribution')
                
                # Analysis type distribution
                ax2.bar(analysis_types.keys(), analysis_types.values(), 
                       color=['#667eea', '#764ba2'])
                ax2.set_title('Analysis Type Distribution')
                ax2.tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.info("No patient data available for statistics")





def create_step_indicator(current_step):
    """Create step indicator for the workflow"""
    steps = [
        {"num": 1, "title": "Data Upload", "icon": "📁"},
        {"num": 2, "title": "Analysis Selection", "icon": "🔬"},
        {"num": 3, "title": "Results Review", "icon": "📊"},
        {"num": 4, "title": "Patient Management", "icon": "🏥"}
    ]
    
    step_html = '<div class="step-container">'
    
    for step in steps:
        if step["num"] < current_step:
            status = "completed"
        elif step["num"] == current_step:
            status = "active"
        else:
            status = "inactive"
            
        step_html += f'''
        <div class="step-item {status}">
            <div class="step-number">{step["icon"]}</div>
            <div class="step-title">{step["title"]}</div>
        </div>
        '''
        
        if step["num"] < len(steps):
            step_html += '<div style="flex: 1; height: 2px; background: #e9ecef; margin: 0 1rem;"></div>'
    
    step_html += '</div>'
    return step_html

def show_dashboard():
    """Show main dashboard with workflow"""
    st.markdown('''
    <div class="main-container">
        <h1>🫁 Lung Cancer Survival Analysis System</h1>
        <p style="font-size: 1.2rem; margin-top: 1rem;">Advanced AI-Powered Medical Data Analysis Platform</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Determine current step
    current_step = 1
    if st.session_state.get('data_uploaded'):
        current_step = 2
    if st.session_state.get('analysis_completed'):
        current_step = 3
    if st.session_state.get('show_patient_management'):
        current_step = 4
    
    # Show step indicator
    st.markdown(create_step_indicator(current_step), unsafe_allow_html=True)
    
    # Main content based on current step
    if current_step == 1:
        show_data_upload()
    elif current_step == 2:
        show_analysis_selection()
    elif current_step == 3:
        show_analysis_results()
    elif current_step == 4:
        show_patient_management()

def show_data_upload():
    """Enhanced data upload interface"""
    st.markdown("## 📁 Step 1: Data Upload")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('''
        <div class="upload-area">
            <h3>📤 Upload Your Dataset</h3>
            <p>Please upload a CSV file containing patient data with the required format</p>
        </div>
        ''', unsafe_allow_html=True)
        
        patient_data = st.file_uploader(
            'Choose your data file',
            type='csv', 
            accept_multiple_files=False,
            help="CSV file should contain columns: name, Status, Time, and feature columns"
        )
        
        # Check if we have uploaded data or already stored data
        dataframe = None
        if patient_data is not None:
            try:
                dataframe = pd.read_csv(patient_data)
                st.session_state['uploaded_data'] = dataframe
                st.markdown('<div class="success-box">✅ Data uploaded successfully!</div>', unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f'<div class="warning-box">❌ Error loading data: {str(e)}</div>', unsafe_allow_html=True)
                return
        elif 'uploaded_data' in st.session_state:
            dataframe = st.session_state['uploaded_data']
            st.markdown('<div class="info-box">📊 Using previously uploaded data</div>', unsafe_allow_html=True)
        
        # If we have data, show data preview and patient selection
        if dataframe is not None:
            # Data preview
            with st.expander("📊 Data Preview", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown('<div class="metric-card"><div class="metric-value">{}</div><div class="metric-label">Total Patients</div></div>'.format(len(dataframe)), unsafe_allow_html=True)
                with col2:
                    st.markdown('<div class="metric-card"><div class="metric-value">{}</div><div class="metric-label">Features</div></div>'.format(len([col for col in dataframe.columns if col not in ['name', 'Status', 'Time']])), unsafe_allow_html=True)
                with col3:
                    st.markdown('<div class="metric-card"><div class="metric-value">{}</div><div class="metric-label">Events</div></div>'.format(dataframe['Status'].sum() if 'Status' in dataframe.columns else 0), unsafe_allow_html=True)
                
                st.dataframe(dataframe.head(10), use_container_width=True)
                
                # Data validation
                required_cols = ['name', 'Status', 'Time']
                missing_cols = [col for col in required_cols if col not in dataframe.columns]
                if missing_cols:
                    st.markdown(f'<div class="warning-box">⚠️ Missing required columns: {missing_cols}</div>', unsafe_allow_html=True)
                    return
                else:
                    st.markdown('<div class="success-box">✅ Data format validation passed</div>', unsafe_allow_html=True)
            
            # Patient selection section
            st.markdown("### 👥 Select Patients for Analysis")
            st.markdown("🔍 **Multi-select patients** - You can select multiple patients for batch analysis")
            
            # Enhanced patient selection with search and filtering
            col_select1, col_select2 = st.columns([3, 1])
            
            # Initialize current selection state
            if 'current_patient_selection' not in st.session_state:
                if 'patients_for_analysis' in st.session_state and st.session_state['patients_for_analysis'] is not None:
                    st.session_state['current_patient_selection'] = st.session_state['patients_for_analysis']['name'].tolist()
                else:
                    st.session_state['current_patient_selection'] = []
            
            with col_select2:
                # Quick selection buttons
                st.markdown("**Quick Select:**")
                if st.button("📋 Select All", use_container_width=True):
                    st.session_state['current_patient_selection'] = dataframe['name'].tolist()
                    st.rerun()
                
                if st.button("🗑️ Clear All", use_container_width=True):
                    st.session_state['current_patient_selection'] = []
                    st.rerun()
                
                # Random selection
                if st.button("🎲 Random 5", use_container_width=True):
                    random_patients = dataframe['name'].sample(min(5, len(dataframe))).tolist()
                    st.session_state['current_patient_selection'] = random_patients
                    st.rerun()
            
            with col_select1:
                selected_patients = st.multiselect(
                    "Choose patients (supports multi-selection):", 
                    options=dataframe['name'].tolist(),
                    default=st.session_state['current_patient_selection'],
                    key="patient_selector",
                    help="Select one or more patients for analysis. You can search by typing patient names."
                )
                
                # Update session state when user manually changes selection
                if selected_patients != st.session_state['current_patient_selection']:
                    st.session_state['current_patient_selection'] = selected_patients
            
            # Debug info (can be removed later)
            if st.checkbox("🔧 Show Debug Info", help="显示调试信息"):
                st.write("**Debug Information:**")
                st.write(f"- session_state['current_patient_selection']: {st.session_state.get('current_patient_selection', 'Not set')}")
                st.write(f"- selected_patients from multiselect: {selected_patients}")
                st.write(f"- dataframe['name'].tolist(): {dataframe['name'].tolist()[:5]}..." if len(dataframe) > 5 else f"- dataframe['name'].tolist(): {dataframe['name'].tolist()}")
            
            # Show selected patients preview
            if selected_patients:
                st.markdown(f"### 📋 Selected Patients ({len(selected_patients)} patients)")
                df_preview = dataframe[dataframe['name'].isin(selected_patients)]
                
                # Display selected patients info
                col_info1, col_info2, col_info3 = st.columns(3)
                with col_info1:
                    st.metric("Selected Patients", len(selected_patients))
                with col_info2:
                    events_count = df_preview['Status'].sum() if 'Status' in df_preview.columns else 0
                    st.metric("Events", events_count)
                with col_info3:
                    event_rate = (events_count / len(selected_patients) * 100) if len(selected_patients) > 0 else 0
                    st.metric("Event Rate", f"{event_rate:.1f}%")
                
                # Show selected patients table
                with st.expander("📊 Preview Selected Patients", expanded=False):
                    display_cols = ['name', 'Status', 'Time']
                    # Add some feature columns if available
                    feature_cols = [col for col in df_preview.columns if col not in ['name', 'Status', 'Time']]
                    if feature_cols:
                        display_cols.extend(feature_cols[:3])  # Show first 3 features
                    
                    st.dataframe(df_preview[display_cols], use_container_width=True)
                
                # Confirmation button
                st.markdown("---")
                col_confirm1, col_confirm2 = st.columns([1, 1])
                
                with col_confirm1:
                    if st.button('✅ Confirm Selection & Proceed', type="primary", use_container_width=True):
                        df_for_analysis = dataframe[dataframe['name'].isin(selected_patients)]
                        st.session_state['patients_for_analysis'] = df_for_analysis
                        st.session_state['data_uploaded'] = True
                        
                        st.success(f"✅ Successfully selected {len(selected_patients)} patients for analysis!")
                        st.markdown('<div class="success-box">🎉 Ready to proceed to Step 2: Analysis Selection</div>', unsafe_allow_html=True)
                        
                        # Auto advance after a short delay
                        time.sleep(1)
                        st.rerun()
                
                with col_confirm2:
                    if st.button('🔄 Modify Selection', use_container_width=True):
                        st.info("👆 Please modify your selection above and confirm again.")
            else:
                st.markdown('<div class="warning-box">⚠️ No patients selected. Please select at least one patient to continue.</div>', unsafe_allow_html=True)
        
        # Show current selection status if data is ready
        if st.session_state.get('patients_for_analysis') is not None:
            st.markdown("---")
            st.markdown("### ✅ Current Selection Status")
            current_df = st.session_state['patients_for_analysis']
            
            col_status1, col_status2, col_status3 = st.columns(3)
            with col_status1:
                st.metric("📊 Patients Ready", len(current_df))
            with col_status2:
                st.metric("📈 Events", current_df['Status'].sum() if 'Status' in current_df.columns else 0)
            with col_status3:
                st.metric("🔬 Features", len([col for col in current_df.columns if col not in ['name', 'Status', 'Time']]))
            
            st.markdown('<div class="success-box">🎯 Data is ready! You can proceed to Step 2: Analysis Selection</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('''
        <div class="info-box">
            <h4>📋 Data Requirements</h4>
            <ul>
                <li><strong>name:</strong> Patient identifier</li>
                <li><strong>Status:</strong> Event occurrence (0/1)</li>
                <li><strong>Time:</strong> Time to event/censoring</li>
                <li><strong>Features:</strong> Clinical/imaging variables</li>
            </ul>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown('''
        <div class="warning-box">
            🔒 <strong>Privacy Notice</strong><br>
            Your data is processed locally and will not be stored permanently. All data is cleared when the session ends.
        </div>
        ''', unsafe_allow_html=True)

def show_analysis_selection():
    """Enhanced analysis selection interface"""
    st.markdown("## 🔬 Step 2: Analysis Selection")
    
    # Check if we have patient data
    if st.session_state.get('patients_for_analysis') is None:
        st.markdown('<div class="warning-box">⚠️ No patient data selected. Please go back to Step 1 and select patients.</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back to Data Upload", use_container_width=True):
                st.session_state['data_uploaded'] = False
                st.rerun()
        with col2:
            if st.button("🔄 Reset Workflow", use_container_width=True):
                # Reset all states
                for key in ['data_uploaded', 'patients_for_analysis', 'uploaded_data', 'current_patient_selection']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        return
    
    st.markdown("### 📊 Choose Analysis Type")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pfs_selected = st.button(
            "📊 Progression-Free Survival (PFS) Analysis",
            use_container_width=True,
            help="Analyze time to disease progression or death"
        )
        
        st.markdown('''
        <div class="analysis-card">
            <h4>📊 PFS Analysis Features:</h4>
            <ul>
                <li>Risk stratification</li>
                <li>Survival probability curves</li>
                <li>Personalized predictions</li>
            </ul>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        os_selected = st.button(
            "📈 Overall Survival (OS) Analysis", 
            use_container_width=True,
            help="Analyze time to death from any cause"
        )
        
        st.markdown('''
        <div class="analysis-card">
            <h4>📈 OS Analysis Features:</h4>
            <ul>
                <li>Mortality risk assessment</li>
                <li>Long-term survival curves</li>
                <li>Feature impact analysis</li>
                <li>Clinical decision support</li>
            </ul>
        </div>
        ''', unsafe_allow_html=True)
    
    if pfs_selected:
        st.session_state['selected_analysis'] = 'PFS'
        st.session_state['analysis_completed'] = False  # Analysis not completed yet, just selected
        st.success("✅ PFS Analysis selected! Proceeding to analysis execution...")
        time.sleep(1)
        st.rerun()
    
    if os_selected:
        st.session_state['selected_analysis'] = 'OS'
        st.session_state['analysis_completed'] = False  # Analysis not completed yet, just selected
        st.success("✅ OS Analysis selected! Proceeding to analysis execution...")
        time.sleep(1)
        st.rerun()
    
    # Show selected patients summary
    if st.session_state.get('patients_for_analysis') is not None:
        st.markdown("---")
        st.markdown("### 👥 Selected Patients Summary")
        df = st.session_state['patients_for_analysis']
        
        # Display summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Patients", len(df))
        with col2:
            events = df['Status'].sum() if 'Status' in df.columns else 0
            st.metric("📈 Events", events)
        with col3:
            event_rate = (events / len(df) * 100) if len(df) > 0 else 0
            st.metric("📋 Event Rate", f"{event_rate:.1f}%")
        with col4:
            features = len([col for col in df.columns if col not in ['name', 'Status', 'Time']])
            st.metric("🔬 Features", features)
        
        # Show patient list in expandable section
        with st.expander("📋 View Selected Patients Details", expanded=False):
            display_cols = ['name', 'Status', 'Time']
            feature_cols = [col for col in df.columns if col not in ['name', 'Status', 'Time']]
            if feature_cols:
                display_cols.extend(feature_cols[:3])  # Show first 3 features
            st.dataframe(df[display_cols], use_container_width=True)
        
        # Option to modify selection
        col_modify1, col_modify2 = st.columns(2)
        with col_modify1:
            if st.button("← Modify Patient Selection", use_container_width=True):
                st.session_state['data_uploaded'] = False
                st.rerun()
        with col_modify2:
            st.markdown("✅ **Selection confirmed** - Choose analysis type above")

def show_analysis_results():
    """Show analysis results with enhanced UI"""
    analysis_type = st.session_state.get('selected_analysis', 'PFS')
    
    st.markdown(f"## 📊 Step 3: {analysis_type} Analysis Results")
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← Back to Analysis Selection"):
            st.session_state['analysis_completed'] = False
            st.rerun()
    with col2:
        if st.button("🔄 Run Analysis Again"):
            st.session_state['analysis_completed'] = False
            st.rerun()
    with col3:
        if st.button("🏥 Patient Management →"):
            st.session_state['show_patient_management'] = True
            st.rerun()
    
    # Load appropriate model and run analysis
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    if analysis_type == 'PFS':
        model_path = os.path.join(current_dir, 'models', 'HRPFS_simple_model.pkl')
        training_data_path = os.path.join(current_dir, 'data', 'Cli_CT_habitat4_PET_habitat1_train_PFS.csv')
        sample_data_path = os.path.join(current_dir, 'data', 'sample_PFS_data.csv')
        analysis_title = "Progression-Free Survival"
    else:
        model_path = os.path.join(current_dir, 'models', 'HROS_simple_model.pkl')
        training_data_path = os.path.join(current_dir, 'data', 'Cli_CT_habitat1_PET_habitat1_train_OS.csv')
        sample_data_path = os.path.join(current_dir, 'data', 'sample_OS_data.csv')
        analysis_title = "Overall Survival"
    
    # Load model
    with st.spinner("🔄 Loading model..."):
        try:
            reconstructor = load_model(model_path)
            train_data = pd.read_csv(training_data_path)
            feature_cols = [col for col in train_data.columns if col not in ['name', 'Status', 'Time']]
            X_train = train_data[feature_cols]
            y_train_time = train_data['Time']
            y_train_event = train_data['Status']
            reconstructor.fit_cox_model(X_train, y_train_time, y_train_event)
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return
    
    # Get data for analysis
    if st.session_state.get('patients_for_analysis') is not None:
        data = st.session_state['patients_for_analysis']
    else:
        st.warning("⚠️ Using sample data for demonstration")
        data = pd.read_csv(sample_data_path)
    
    # Continue with the rest of the analysis code...
    # [The rest of your analysis code goes here, but with enhanced UI elements]

def show_patient_management():
    """Enhanced patient management interface"""
    st.markdown("## 🏥 Step 4: Patient Management")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back to Results"):
            st.session_state['show_patient_management'] = False
            st.rerun()
    with col2:
        if st.button("🔄 New Analysis"):
            # Reset all session state
            for key in ['data_uploaded', 'analysis_completed', 'show_patient_management', 'selected_analysis']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    # Patient management interface (use your existing patient_management_interface function)
    patient_management_interface()

def show_workflow_dashboard():
    """Show the main workflow dashboard"""
    # Determine current step
    current_step = 1
    if st.session_state.get('data_uploaded'):
        current_step = 2
    if st.session_state.get('selected_analysis'):
        current_step = 3
    if st.session_state.get('analysis_completed'):
        current_step = 4
    
    # Show step indicator
    st.markdown(create_step_indicator(current_step), unsafe_allow_html=True)
    
    # Main content based on current step
    if current_step == 1:
        show_data_upload()
    elif current_step == 2:
        show_analysis_selection()
    elif current_step == 3:
        run_selected_analysis()
    elif current_step == 4:
        display_analysis_results()

def run_selected_analysis():
    """Run the selected analysis type"""
    analysis_type = st.session_state.get('selected_analysis', 'PFS')
    
    st.markdown(f"## 🔬 Step 3: Running {analysis_type} Analysis")
    
    # Navigation buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back to Analysis Selection"):
            st.session_state['selected_analysis'] = None
            st.rerun()
    with col2:
        if st.button("🔄 Change Analysis Type"):
            st.session_state['selected_analysis'] = None
            st.rerun()
    
    # Load appropriate model and run analysis
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    if analysis_type == 'PFS':
        model_path = os.path.join(current_dir, 'models', 'HRPFS_simple_model.pkl')
        training_data_path = os.path.join(current_dir, 'data', 'Cli_CT_habitat4_PET_habitat1_train_PFS.csv')
        sample_data_path = os.path.join(current_dir, 'data', 'sample_PFS_data.csv')
        analysis_title = "Progression-Free Survival"
    else:
        model_path = os.path.join(current_dir, 'models', 'HROS_simple_model.pkl')
        training_data_path = os.path.join(current_dir, 'data', 'Cli_CT_habitat1_PET_habitat1_train_OS.csv')
        sample_data_path = os.path.join(current_dir, 'data', 'sample_OS_data.csv')
        analysis_title = "Overall Survival"
    
    # Get data for analysis
    if st.session_state.get('patients_for_analysis') is not None:
        data = st.session_state['patients_for_analysis']
        st.success(f"✅ Using uploaded data with {len(data)} patients")
    else:
        st.warning("⚠️ No uploaded data found. Using sample data for demonstration.")
        data = pd.read_csv(sample_data_path)
    
    # Load model with progress indication
    with st.status(f"🔄 Loading {analysis_type} model and preparing analysis...", expanded=True) as status:
        try:
            st.write("Loading model package...")
            reconstructor = load_model(model_path)
            
            st.write("Loading training data...")
            train_data = pd.read_csv(training_data_path)
            feature_cols = [col for col in train_data.columns if col not in ['name', 'Status', 'Time']]
            X_train = train_data[feature_cols]
            y_train_time = train_data['Time']
            y_train_event = train_data['Status']
            
            st.write("Reconstructing Cox model...")
            reconstructor.fit_cox_model(X_train, y_train_time, y_train_event)
            
            st.write("Running analysis on patient data...")
            # Run the actual analysis
            patient_ids = data.index
            patient_names = data['name']
            X = data[feature_cols]
            
            # Calculate risk scores
            risk_results = reconstructor.calculate_risk_score(X, return_percentile=True)
            risk_scores = risk_results['risk_scores']
            risk_groups = risk_results['risk_groups']
            risk_percentiles = risk_results['risk_percentiles']
            
            # Predict survival curves
            time_points = list(range(3, 61, 3))
            survival_probs = reconstructor.predict_survival_function(X, time_points=time_points)
            

            
            # Store results in session state
            st.session_state['analysis_results'] = {
                'analysis_type': analysis_type,
                'analysis_title': analysis_title,
                'patient_ids': patient_ids,
                'patient_names': patient_names,
                'risk_scores': risk_scores,
                'risk_groups': risk_groups,
                'risk_percentiles': risk_percentiles,
                'survival_probs': survival_probs,
                'time_points': time_points,
                'feature_cols': feature_cols,
                'X': X,
            }
            
            st.session_state['analysis_completed'] = True
            status.update(label="✅ Analysis completed successfully!", state="complete", expanded=False)
            
        except Exception as e:
            error_msg = str(e)
            # Check if this is a feature mismatch error (wrong model for data type)
            if "not in index" in error_msg or "KeyError" in str(type(e)):
                st.error("❌ Error during analysis. Choose the Right Analysing Model!")
                st.warning("💡 **Tip:** Make sure you've selected the correct analysis type for your data:")
                st.markdown("""
                - **PFS Analysis**: Use for Progression-Free Survival data
                - **OS Analysis**: Use for Overall Survival data
                
                The uploaded data might be designed for a different analysis type.
                """)
                
                # Suggest switching analysis type
                col_switch1, col_switch2 = st.columns(2)
                with col_switch1:
                    if st.button("🔄 Try OS Analysis Instead", use_container_width=True):
                        st.session_state['selected_analysis'] = 'OS'
                        st.rerun()
                with col_switch2:
                    if st.button("🔄 Try PFS Analysis Instead", use_container_width=True):
                        st.session_state['selected_analysis'] = 'PFS'
                        st.rerun()
            else:
                st.error(f"❌ Error during analysis: {error_msg}")
            return
    
    # Auto-advance to results
    if st.session_state.get('analysis_completed'):
        st.success("🎉 Analysis completed! Proceeding to results...")
        time.sleep(1)
        st.rerun()

def display_analysis_results():
    """Display comprehensive analysis results"""
    if 'analysis_results' not in st.session_state:
        st.error("No analysis results found. Please run analysis first.")
        return
        
    results = st.session_state['analysis_results']
    analysis_type = results['analysis_type']
    analysis_title = results['analysis_title']
    
    st.markdown(f"## 📊 Step 4: {analysis_title} Analysis Results")
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← Back to Analysis"):
            st.session_state['analysis_completed'] = False
            st.rerun()
    with col2:
        if st.button("🔄 Run New Analysis"):
            # Reset analysis state
            st.session_state['analysis_completed'] = False
            st.session_state['selected_analysis'] = None
            if 'analysis_results' in st.session_state:
                del st.session_state['analysis_results']
            st.rerun()
    with col3:
        if st.button("🏥 Manage Patients"):
            st.session_state['current_page'] = 'patient_management'
            st.rerun()
    
    # Display results summary
    st.markdown(f"""
    <div class="main-container">
        <h2>📊 {analysis_title} Analysis Complete</h2>
        <p>Analyzed {len(results['patient_names'])} patients with comprehensive risk assessment</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Results summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-value">{}</div><div class="metric-label">Total Patients</div></div>'.format(len(results['patient_names'])), unsafe_allow_html=True)
    with col2:
        high_risk_count = sum(1 for group in results['risk_groups'] if group == 'High Risk')
        st.markdown('<div class="metric-card"><div class="metric-value">{}</div><div class="metric-label">High Risk</div></div>'.format(high_risk_count), unsafe_allow_html=True)
    with col3:
        low_risk_count = sum(1 for group in results['risk_groups'] if group == 'Low Risk')
        st.markdown('<div class="metric-card"><div class="metric-value">{}</div><div class="metric-label">Low Risk</div></div>'.format(low_risk_count), unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><div class="metric-value">{}</div><div class="metric-label">Features</div></div>'.format(len(results['feature_cols'])), unsafe_allow_html=True)
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["👤 Individual Reports", "📊 Summary Charts", "📋 Patient Management"])
    
    with tab1:
        show_individual_patient_reports(results)
    
    with tab2:
        show_summary_charts(results)
        
    with tab3:
        show_patient_management_options(results)

def show_individual_patient_reports(results):
    """Show individual patient analysis reports"""
    st.markdown("### 👤 Individual Patient Analysis Reports")
    
    for i, patient_id in enumerate(results['patient_ids']):
        patient_name = results['patient_names'].iloc[i]
        risk_color = "🔴" if results['risk_groups'][i] == "High Risk" else "🟢"
        
        with st.expander(f"{risk_color} {patient_name} - Detailed Analysis", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                # Risk assessment table
                st.markdown("#### 🎯 Risk Assessment")
                risk_data = pd.DataFrame({
                    "Metric": ["Risk Score", "Risk Percentile", "Risk Group"],
                    "Value": [
                        f"{results['risk_scores'][i]:.4f}",
                        f"{results['risk_percentiles'][i]:.1f}%",
                        results['risk_groups'][i]
                    ]
                })
                st.table(risk_data.set_index("Metric"))
            
            with col2:
                # Survival probabilities
                st.markdown(f"#### 📊 {results['analysis_title']} Probabilities")
                survival_times = [6, 12, 24, 36, 60]
                survival_data = []
                
                for j, t in enumerate(survival_times):
                    if j * 2 < len(results['survival_probs'][i]):  # Adjust index for time points
                        prob = results['survival_probs'][i][j * 2]
                        survival_data.append({
                            "Time": f"{t} months",
                            "Probability": f"{prob:.3f} ({prob * 100:.1f}%)"
                        })
                
                if survival_data:
                    survival_df = pd.DataFrame(survival_data)
                    st.table(survival_df.set_index("Time"))
            
            # Individual charts
            col3, col4 = st.columns(2)
            
            with col3:
                # Individual survival curve
                survival_fig = generate_individual_survival_curve(results['survival_probs'][i], results['time_points'], patient_id+1)
                st.pyplot(survival_fig)
                plt.close(survival_fig)
            
            with col4:
                # Individual risk plot
                risk_fig = generate_individual_risk_plot(results['risk_scores'][i], results['risk_groups'][i], patient_id+1, results['risk_percentiles'][i])
                st.pyplot(risk_fig)
                plt.close(risk_fig)

def show_summary_charts(results):
    """Show summary analysis charts"""
    st.markdown("### 📊 Overall Analysis Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Risk Score Distribution")
        risk_fig = plot_risk_scores(results['risk_scores'], results['risk_groups'], results['patient_ids'])
        st.pyplot(risk_fig)
        plt.close(risk_fig)
    
    with col2:
        st.markdown("#### 📈 Survival Probability Curves")
        survival_fig = plot_survival_curves(results['survival_probs'], results['time_points'], results['patient_ids'])
        st.pyplot(survival_fig)
        plt.close(survival_fig)



def show_patient_management_options(results):
    """Show patient management options"""
    st.markdown("### 📋 Add Patients to Management System")
    
    # Select patients to add
    patient_options = []
    for i, patient_id in enumerate(results['patient_ids']):
        patient_name = results['patient_names'].iloc[i]
        risk_group = results['risk_groups'][i]
        patient_options.append(f"{patient_name} ({risk_group})")
    
    selected_indices = st.multiselect(
        "Select patients to add to management system:",
        options=list(range(len(patient_options))),
        format_func=lambda x: patient_options[x],
        help="Selected patients will be added to the patient management system"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📋 Add Selected Patients", type="primary"):
            if selected_indices:
                add_patients_to_management(results, selected_indices)
            else:
                st.warning("Please select patients to add")
    
    with col2:
        if st.button("📋 Add All Patients"):
            add_patients_to_management(results, list(range(len(results['patient_ids']))))

def add_patients_to_management(results, indices):
    """Add selected patients to management system"""
    if 'patient_management' not in st.session_state:
        st.session_state['patient_management'] = {}
    
    analysis_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    added_count = 0
    
    for idx in indices:
        patient_id = str(uuid.uuid4())[:8]
        patient_info = {
            'id': patient_id,
            'name': results['patient_names'].iloc[idx],
            'risk_score': f"{results['risk_scores'][idx]:.4f}",
            'risk_group': results['risk_groups'][idx],
            'risk_percentile': f"{results['risk_percentiles'][idx]:.1f}",
            'analysis_type': results['analysis_title'],
            'analysis_time': analysis_time,
            'patient_index': int(results['patient_ids'][idx])
        }
        
        # Check if already exists
        existing_patients = [p for p in st.session_state['patient_management'].values() 
                           if p['name'] == patient_info['name'] and p['analysis_type'] == results['analysis_title']]
        
        if not existing_patients:
            st.session_state['patient_management'][patient_id] = patient_info
            added_count += 1
    
    if added_count > 0:
        st.success(f"✅ Successfully added {added_count} patients to management system!")
        st.info("💡 You can access the patient management system from the sidebar.")
    else:
        st.warning("⚠️ Selected patients are already in the management system.")

def main():
    """Main application function with enhanced workflow"""
    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = 'workflow'
    if 'data_uploaded' not in st.session_state:
        st.session_state['data_uploaded'] = False
    if 'analysis_completed' not in st.session_state:
        st.session_state['analysis_completed'] = False
    if 'selected_analysis' not in st.session_state:
        st.session_state['selected_analysis'] = None
    if 'patients_for_analysis' not in st.session_state:
        st.session_state['patients_for_analysis'] = None
    if 'patients_with_analysis' not in st.session_state:
        st.session_state['patients_with_analysis'] = []
    if 'patient_management' not in st.session_state:
        st.session_state['patient_management'] = {}
    
    # Main title
    st.markdown('''
    <div class="main-container">
        <h1>🫁 Lung Cancer Survival Analysis System</h1>
        <p style="font-size: 1.2rem; margin-top: 1rem;">Advanced AI-Powered Medical Data Analysis Platform</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown('<div class="sidebar-title">🫁 Navigation</div>', unsafe_allow_html=True)
        
        # Main navigation
        page_choice = st.radio(
            "Choose Section:",
            ["🔄 Analysis Workflow", "🏥 Patient Management"],
            index=0 if st.session_state['current_page'] == 'workflow' else 1,
            help="Analysis Workflow: Step-by-step analysis process\nPatient Management: Manage analyzed patients"
        )
        
        # Update page state
        if page_choice == "🔄 Analysis Workflow":
            st.session_state['current_page'] = 'workflow'
        else:
            st.session_state['current_page'] = 'patient_management'
        
        st.markdown("---")
        
        # Quick actions
        if st.button("🏠 Reset Workflow", use_container_width=True):
            # Reset workflow state
            for key in ['data_uploaded', 'analysis_completed', 'selected_analysis', 'patients_for_analysis', 'uploaded_data', 'current_patient_selection']:
                if key in st.session_state:
                    if 'completed' in key or 'uploaded' in key:
                        st.session_state[key] = False
                    else:
                        del st.session_state[key]
            if 'analysis_results' in st.session_state:
                del st.session_state['analysis_results']
            st.session_state['current_page'] = 'workflow'
            st.rerun()
        
        st.markdown("---")
        
        # Status display
        st.markdown("### 📊 Current Status")
        
        status_items = [
            ("Data Uploaded", st.session_state.get('data_uploaded', False)),
            ("Analysis Selected", st.session_state.get('selected_analysis') is not None),
            ("Analysis Completed", st.session_state.get('analysis_completed', False)),
        ]
        
        for item, status in status_items:
            icon = "✅" if status else "⏳"
            color = "#10ac84" if status else "#6c757d"
            st.markdown(f'<span style="color: {color}">{icon} {item}</span>', unsafe_allow_html=True)
        
        # Quick stats
        if st.session_state.get('patients_for_analysis') is not None:
            st.markdown("---")
            st.markdown("### 📈 Analysis Stats")
            df = st.session_state['patients_for_analysis']
            st.metric("Selected Patients", len(df))
            if 'Status' in df.columns:
                st.metric("Events", df['Status'].sum())
        
        if st.session_state.get('patient_management'):
            st.metric("Managed Patients", len(st.session_state['patient_management']))
    
    # Main content based on current page
    if st.session_state['current_page'] == 'patient_management':
        patient_management_interface()
    else:
        show_workflow_dashboard()

if __name__ == "__main__":
    main()


