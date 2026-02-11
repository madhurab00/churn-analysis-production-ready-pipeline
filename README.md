# Building Production Ready Machine Learning System

This project demonstrates production-ready machine learning pipelines with comprehensive MLflow artifact tracking, focusing on customer churn prediction.

## 🎯 Project Overview

A complete ML system with enhanced MLflow tracking that provides:
- **Comprehensive Data Lineage**
- **Rich Artifact Management**
- **Production-Ready Monitoring**
- **Complete Reproducibility**

This repository implements a **production-ready machine learning pipeline** for customer churn prediction with comprehensive MLOps practices. The project demonstrates enterprise-grade ML engineering with:

- **Dual Implementation**: Scikit-learn (standard) and PySpark (distributed) pipelines
- **Orchestration**: Apache Airflow for workflow management
- **Experiment Tracking**: MLflow for versioning, metrics, and model registry
- **Observability**: Enhanced logging, monitoring, and data lineage tracking
- **Scalability**: Distributed processing with PySpark for big data scenarios
- **Reproducibility**: Version-controlled experiments and automated pipelines

---

## 📂 Repository Structure

```
production-ready-ml-pipeline/
├── README.md                          # Project documentation
├── Makefile                           # Automation commands
├── config.yaml                        # Configuration parameters
├── requirements.txt                   # Python dependencies
│
├── .airflow/                          # Airflow home directory
│   ├── airflow.cfg                    # Airflow configuration
│   ├── airflow.db                     # Metadata database
│   ├── dags/                          # DAG definitions
│   │   ├── data_pipeline_dag.py
│   │   └── train_pipeline_dag.py
│   └── logs/                          # Airflow execution logs
│
├── artifacts/                         # Training artifacts
│   ├── data/                          # Processed datasets
│   │   ├── X_train.csv
│   │   ├── X_test.csv
│   │   ├── Y_train.csv
│   │   └── Y_test.csv
│   ├── encode/                        # Feature encoders
│   │   ├── Gender_encoder.json
│   │   └── Geography_encoder.json
│   ├── models/                        # Trained models
│   │   └── churn_analysis.joblib
│   └── mlflow_run_artifacts/          # MLflow artifacts by run
│       └── {run_id}/
│           ├── visualizations/
│           └── final_csv_files/
│
├── data/                              # Data storage
│   ├── raw/                           # Original datasets
│   │   └── ChurnModelling.csv
│   └── processed/                     # Cleaned datasets
│       └── imputed.csv
│
├── mlruns/                            # MLflow tracking store
│   ├── 0/                             # Default experiment
│   ├── models/                        # Model registry
│   └── {experiment_id}/               # Experiment runs
│       └── {run_id}/
│           ├── artifacts/
│           ├── metrics/
│           ├── params/
│           └── tags/
│
├── pipelines/                         # Pipeline orchestration
│   ├── data_pipeline.py               # Data preprocessing
│   ├── training_pipeline.py           # Model training
│   └── streaming_inference_pipeline.py # Batch inference
│
├── src/                               # Core modules
│   ├── data_ingestion.py              # Data loading
│   ├── data_splitter.py               # Train/test split
│   ├── feature_binning.py             # Feature discretization
│   ├── feature_encoding.py            # Categorical encoding
│   ├── feature_scaling.py             # Normalization
│   ├── handle_missing_values.py       # Imputation
│   ├── model_building.py              # Model architecture
│   ├── model_evaluation.py            # Performance metrics
│   ├── model_inference.py             # Predictions
│   ├── model_training.py              # Training logic
│   └── outlier_detection.py           # Anomaly detection
│
└── utils/                             # Helper functions
    ├── airflow_tasks.py               # Airflow task definitions
    ├── config.py                      # Config management
    └── mlflow_utils.py                # MLflow helpers
```

---

## ✨ Key Features

### 1. 📊 Enhanced Data Pipeline
- **Stage-wise Data Profiling**: Track data quality at each transformation step
  - Raw data → Missing value handling → Encoding → Scaling → Final
- **Automatic Visualizations**: 
  - Feature distributions (histograms, box plots)
  - Correlation heatmaps
  - Missing value patterns
- **Data Lineage Tracking**: Full traceability using MLflow datasets
- **Quality Metrics**: 
  - Row/column counts
  - Missing value percentages
  - Memory usage
  - Data drift detection
- **Error Handling**: Comprehensive logging and failure recovery

### 2. 🎓 Enhanced Training Pipeline
- **Model Performance Tracking**:
  - Confusion matrices
  - Accuracy
  - Precision 
  - Recall 
  - F1-Score
- **Training Metrics**:
  - Training time
  - Model size
  - Hyperparameters
  - Cross-validation scores
- **Model Registry**: Versioned models with metadata
- **MLflow Integration**: Full experiment reproducibility

### 3. 🔮 Enhanced Inference Pipeline
- **Batch Prediction Tracking**: Monitor inference jobs
- **Performance Monitoring**:
  - Inference time per batch
  - Prediction distribution
  - Confidence scores
- **Logging**: Predictions and metrics logged to MLflow
- **Model Serving**: Ready for deployment integration

### 4. 🔬 MLflow Integration
- **Experiment Tracking**: Parameters, metrics, artifacts
- **Model Versioning**: Automatic model registry
- **Dataset Tracking**: Input/output data lineage
- **Artifact Management**: Organized storage by run ID
- **Visualization**: Interactive plots and dashboards

---

## 🛠️ Prerequisites

### System Requirements
- **Python**: ≥ 3.10
- **Java**: ≥ 8 (required for PySpark)
- **Memory**: ≥ 8GB RAM recommended
- **Storage**: ≥ 5GB free space

### Required Tools
```bash
python3 --version  # Python 3.10+
java -version      # Java 8+
pip --version      # Latest pip
git --version      # Git for version control
```

### Python Packages
- **ML/Data**: scikit-learn, pandas, numpy
- **Big Data**: pyspark ≥ 3.x
- **MLOps**: mlflow ≥ 2.x
- **Orchestration**: apache-airflow ≥ 2.x
- **Visualization**: matplotlib, seaborn

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/madhurabe00/production-ready-ml-pipeline.git
cd "Production ready pipeline/pyspark"
```

### 2. Set Up Environment

#### Linux/WSL (Recommended for Airflow)
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

#### Windows (Standard Pipeline Only)
```powershell
# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Initialize Airflow (Optional - for PySpark pipelines)
```bash
# Set Airflow home directory
export AIRFLOW_HOME=$(pwd)/.airflow

# Initialize database
airflow db init

# Create admin user
airflow users create \
    --username admin \
    --password admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com

# Create DAGs folder
mkdir -p .airflow/dags
```

### 4. Configure the Project
Edit `config.yaml` to customize:
```yaml
data:
  raw_path: "data/raw/churndataset.csv"
  processed_path: "data/processed/"

model:
  name: "RandomForestClassifier"
  hyperparameters:
    n_estimators: 100
    max_depth: 10

mlflow:
  tracking_uri: "file:./mlruns"
  experiment_name: "churn_prediction"
```

---

## 🎮 Running the Pipelines

### Option 1: Using Makefile (Recommended)

```bash
# Run data pipeline
make data-pipeline

# Train model
make train-pipeline

# Run inference
make streaming-inference

# Start MLflow UI
make mlflow-ui

# Start Airflow (if configured)
make airflow-start

# Run all pipelines in sequence
run-all

# Stop all running MLflow servers
stop-all
```

### Option 2: Direct Execution

#### Data Pipeline
```bash
# Activate environment
source .venv/bin/activate  # Linux/WSL
# OR
.\.venv\Scripts\Activate.ps1  # Windows

# Change directory to scikit implemetation or spark implementation

cd ./scikit 
# OR
cd ./pyspark

# Run data pipeline
python pipelines/data_pipeline.py
```

#### Training Pipeline
```bash
python pipelines/training_pipeline.py
```

#### Inference Pipeline
```bash
python pipelines/streaming_inference_pipeline.py
```

### Option 3: Using Airflow (Distributed Processing)

```bash
# Start Airflow webserver (Terminal 1)
export AIRFLOW_HOME=$(pwd)/.airflow
airflow webserver -p 8080

# Start Airflow scheduler (Terminal 2)
export AIRFLOW_HOME=$(pwd)/.airflow
airflow scheduler

# OR
airflow standalone

# Access Airflow UI
# URL: http://localhost:8080
# Username: admin
# Password: admin

# Enable and trigger DAGs through the UI
```

---

## 📊 Monitoring and Visualization

### MLflow Tracking UI
```bash
# Start MLflow UI
mlflow ui --port 5000

# Access at: http://localhost:5000
```

**Features**:
- Compare experiments and runs
- View metrics over time
- Download artifacts
- Register models
- Track data lineage

### Airflow UI
```bash
# Access at: http://localhost:8080
```

**Features**:
- Monitor DAG runs
- View task logs
- Trigger manual runs
- Configure schedules
- Check task dependencies

---

## 📈 Model Performance

### Current Best Model
- **Algorithm**: Random Forest Classifier
- **Accuracy**: 0.776
- **Precision**: 0.767
- **Recall**: 0.763
- **F1-Score**: 0.776
- **ROC-AUC**: 0.89

---

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Update documentation
6. Submit a pull request

### Code Style
- Follow PEP 8
- Use type hints
- Add docstrings
- Write unit tests

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

- **Madhura Edirisooriya** - Initial work - [GitHub](https://github.com/madhurab00)

---

## 🙏 Acknowledgments

- Isuru Alagiyawanna(Machine Learning Zuu)
- Apache Airflow community
- MLflow contributors
- PySpark documentation
- Scikit-learn team

---

## 📚 Additional Resources
- [Zuu Crew.ai](https://www.zuucrew.ai)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [PySpark Documentation](https://spark.apache.org/docs/latest/api/python/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

---

## 🔗 Related Projects

- [MLOps Best Practices](https://github.com/topics/mlops)
- [Production ML Systems](https://github.com/topics/production-ml)
- [Data Pipeline Examples](https://github.com/topics/data-pipeline)

---

**⭐ If you find this project helpful, please consider giving it a star!**
