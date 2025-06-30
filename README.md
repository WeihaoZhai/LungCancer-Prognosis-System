# Lung Cancer Prognosis Prediction System

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive web-based platform for lung cancer prognosis prediction using machine learning and multimodal radiomics features.

## 🎯 Overview

This system integrates clinical features with multimodal radiomics features from PET and CT imaging to predict both Overall Survival (OS) and Progression-Free Survival (PFS) in lung cancer patients. The platform provides real-time risk assessment, personalized survival curve prediction, and comprehensive patient management capabilities.

## ✨ Key Features

- **Dual Prediction Models**: Separate models for OS and PFS prediction
- **Multimodal Radiomics**: Integration of PET and CT radiomics features with clinical data
- **Real-time Prediction**: Sub-second prediction latency for individual patients
- **Interactive Web Interface**: Modern, user-friendly Streamlit-based interface
- **Comprehensive Visualization**: Personalized survival curves and risk assessment plots
- **Patient Management**: Persistent patient data storage and retrieval system
- **Batch Processing**: Support for multiple patient analysis

## 🏗️ System Architecture

```
LungCancer-Prognosis-System/
├── src/                    # Source code
│   └── app.py             # Main Streamlit application
├── models/                # Pre-trained model files
│   ├── HROS_simple_model.pkl
│   └── HRPFS_simple_model.pkl
├── data/                  # Dataset files
│   ├── training data (OS/PFS)
│   └── sample data
├── docs/                  # Documentation
├── assets/               # Static assets
└── requirements.txt      # Dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/LungCancer-Prognosis-System.git
cd LungCancer-Prognosis-System
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run src/app.py
```

4. Open your browser and navigate to `http://localhost:8501`

## 📊 Model Features

### OS Prediction Model (25 features)
- **Clinical Features**: Age, clinical staging
- **CT Radiomics**: Habitat1 morphological and texture features
- **PET Radiomics**: Habitat1 metabolic and texture features

### PFS Prediction Model (26 features)
- **PET Parameters**: TLG, MTV, SUVmax
- **Clinical Features**: Gender, pathological staging
- **CT Radiomics**: Habitat4 region features
- **PET Radiomics**: Habitat1 high-order texture features

## 🛠️ Technical Implementation

- **Framework**: Streamlit for web interface
- **ML Backend**: Scikit-learn, Lifelines (Cox regression)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Model Architecture**: Cox Proportional Hazards Model with Elastic Net regularization

## 📈 Performance Metrics

- **Prediction Latency**: <1 second per patient
- **Batch Processing**: 10 patients in <5 seconds
- **Model Evaluation**: C-index for concordance assessment
- **Risk Stratification**: ROC-based threshold optimization

## 💻 Usage

### Data Input
- Upload patient data in CSV format
- Ensure all required features are included
- Use provided sample data for testing

### Analysis Workflow
1. **Data Upload**: Upload patient dataset or use sample data
2. **Analysis Selection**: Choose OS or PFS prediction
3. **Model Execution**: Automatic feature processing and prediction
4. **Results Visualization**: View individual reports and summary charts
5. **Patient Management**: Save results and compare across patients

### Output Features
- Individual risk scores and percentile rankings
- Survival probability predictions (6, 12, 24, 36, 60 months)
- Risk stratification (High/Low risk classification)
- Interactive survival curves and risk distribution plots

## 📚 Documentation

- [Installation Guide](docs/installation.md)
- [User Manual](docs/user_guide.md)
- [API Documentation](docs/api.md)
- [Model Details](docs/model_details.md)

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📖 Citation

If you use this software in your research, please cite:

```bibtex
@article{lungcancer_prognosis_2024,
  title={Web-based Lung Cancer Prognosis Prediction System Using Multimodal Radiomics and Machine Learning},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

## 🔬 Research Background

This system was developed as part of research into personalized cancer prognosis prediction using advanced radiomics analysis and machine learning techniques. The models were trained on a cohort of 318 lung cancer patients with comprehensive clinical and imaging data.

## ⚠️ Disclaimer

This software is for research purposes only and should not be used for clinical decision-making without appropriate validation and regulatory approval.

## 📞 Contact

For questions or support, please contact [your.email@institution.edu](mailto:your.email@institution.edu)

---

**Keywords**: Lung Cancer, Prognosis Prediction, Radiomics, Machine Learning, Cox Regression, Survival Analysis, Web Application 