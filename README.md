# ⚽ EPL Foul Win Probability Model

A comprehensive **Machine Learning-based Sports Analytics application** designed to predict the probability that a player will **win a foul after receiving or recovering possession** during an English Premier League (EPL) match. The project leverages **event-level football data, predictive modeling, feature engineering, and statistical analysis** to transform raw match events into actionable insights for player and team performance evaluation.

This project simulates a real-world **Sports Analytics and Decision Intelligence platform**, demonstrating how event-stream football data can be transformed into predictive models for tactical analysis, player evaluation, opponent scouting, and performance optimization.

The project follows a complete **Machine Learning lifecycle**, including data ingestion, exploratory data analysis, target variable creation, feature engineering, model training, model evaluation, testing, version control, and CI/CD automation to deliver a reproducible, production-ready analytics pipeline.

---

# 📌 Project Status

> ⚙️ **In Progress**

Current Progress

- ✅ Phase 1 – Data Understanding
- ✅ Phase 2 – Target Variable Creation
- 🚧 Phase 3 – Feature Engineering
- ⏳ Phase 4 – Model Training
- ⏳ Phase 5 – Model Evaluation

---

# 🎯 Problem Statement

Given event-level football data from English Premier League matches, predict whether a player who **receives** or **recovers** the ball will subsequently **win a foul during the same possession**.

This is formulated as a **binary classification** problem.

Target Variable

| Value | Meaning |
|--------|----------|
| 1 | Player wins a foul later in the same possession |
| 0 | Player does not win a foul |

---

# 📂 Dataset

The project uses **StatsBomb Open Data** event datasets.

Datasets include:

- EPL Event Data
- EPL Match Data

Dataset Size

| Dataset | Records |
|----------|---------|
| Events | 1,313,783 |
| Matches | 380 |

---

# 🚀 Features

## 📊 Data Analysis

- Exploratory Data Analysis (EDA)
- Missing Value Analysis
- Event Distribution Analysis
- Possession Analysis
- Dataset Validation

---

## ⚽ Target Creation

- Ball Receipt* detection
- Ball Recovery detection
- Same-possession event tracking
- Binary target generation
- Automated target pipeline

---

## 🧠 Feature Engineering

- Spatial Features
- Temporal Features
- Possession Features
- Match Context Features
- Pressure Features
- Event Sequence Features

---

## 🤖 Machine Learning

Models to be implemented

- Logistic Regression
- Random Forest
- XGBoost
- Gradient Boosting

Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC
- Confusion Matrix

---

## 🧪 Testing

- PyTest Unit Tests
- Data Validation
- Pipeline Testing

---

## 🔄 CI/CD

GitHub Actions

- Automated Testing
- Continuous Integration
- Multi-branch Workflow

---

# 🏗️ Project Structure

```text
epl-foul-prediction/
│
├── .github/
│   └── workflows/
│       └── main.yml
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_target_creation.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_model_evaluation.ipynb
│
├── outputs/
│
├── src/
│   ├── config.py
│   ├── logger.py
│   ├── data_loader.py
│   ├── target_creation.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── utils.py
│
├── tests/
│   ├── test_data_loader.py
│   ├── test_target_creation.py
│   └── test_feature_engineering.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/nagasantoshchavvakula/epl-foul-win-probability-model.git

cd epl-foul-win-probability-model
```

Create Virtual Environment

```bash
python -m venv venv
```

Activate Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Project

Run notebooks sequentially

```text
01_data_understanding.ipynb

↓

02_target_creation.ipynb

↓

03_feature_engineering.ipynb

↓

04_model_training.ipynb

↓

05_model_evaluation.ipynb
```

---

# 🧪 Run Tests

Run all tests

```bash
pytest
```

Verbose mode

```bash
pytest -v
```

Coverage

```bash
pytest --cov=src
```

---

# 📊 Machine Learning Pipeline

```text
Raw Event Data
        │
        ▼
Data Loading
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Target Creation
        │
        ▼
Feature Engineering
        │
        ▼
Train/Test Split
        │
        ▼
Model Training
        │
        ▼
Hyperparameter Tuning
        │
        ▼
Model Evaluation
        │
        ▼
Prediction
```

---

# 🛠️ Technologies Used

## Programming

- Python

## Data Processing

- Pandas
- NumPy

## Machine Learning

- Scikit-learn
- XGBoost

## Visualization

- Matplotlib
- Seaborn

## Development

- Jupyter Notebook
- PyTest

## DevOps

- Git
- GitHub
- GitHub Actions

---

# 💡 Skills Demonstrated

### Machine Learning

- Binary Classification
- Feature Engineering
- Model Evaluation
- Predictive Analytics

### Sports Analytics

- Football Event Analysis
- Possession Analytics
- Player Performance Analytics

### Data Science

- Exploratory Data Analysis
- Statistical Analysis
- Data Visualization

### Software Engineering

- Modular Architecture
- Logging
- Configuration Management
- Unit Testing

### MLOps

- CI/CD
- GitHub Actions
- Git Workflow
- Reproducible Pipelines

---

# 📈 Current Results

Current dataset generated

| Metric | Value |
|---------|-------|
| Candidate Events | 381,267 |
| Positive Samples | 8,686 |
| Negative Samples | 372,581 |
| Positive Rate | 2.28% |

---

# 🚀 Future Improvements

- Hyperparameter Optimization
- Cross Validation
- SHAP Explainability
- Model Calibration
- Probability Calibration
- MLflow Experiment Tracking
- Docker Deployment
- REST API
- Interactive Dashboard

---

# 🤝 Contributing

Contributions, feature requests, and suggestions are welcome.

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to your branch
5. Open a Pull Request

---

# 📜 License

This project is intended for educational and portfolio purposes.

---

# 👨‍💻 Author

**Nagasantosh Chavvakula**

- LinkedIn: https://www.linkedin.com/in/nagasantoshchavvakula
- GitHub: https://github.com/nagasantoshchavvakula

---