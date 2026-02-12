# RetainX -- A Data-Driven Approach to Customer Churn Prediction

## 📌 Project Overview

RetainX is an end-to-end Machine Learning system designed to predict
customer churn and generate actionable business insights. The system
integrates data preprocessing, ML model training, REST API deployment,
analytics generation, and automated PDF reporting into a
production-style application.

------------------------------------------------------------------------

## 🏗 System Architecture

``` mermaid
flowchart TD
    A[Raw Dataset - Telco Customer Churn CSV] --> B[Data Cleaning & Preprocessing]
    B --> C[Feature Engineering & Encoding]
    C --> D[Model Training - Scikit-learn / XGBoost]
    D --> E[Model Serialization - Model.sav]

    E --> F[Flask Backend API]
    F --> G[/predict Endpoint]
    F --> H[/analytics Endpoint]
    F --> I[/generate-pdf Endpoint]

    G --> J[Churn Prediction + Probability]
    H --> K[Business Analytics Dashboard Data]
    I --> L[Automated PDF Report Generation]

    F --> M[Frontend UI - HTML/CSS/JS]
```

------------------------------------------------------------------------

## ⚙️ Tech Stack

-   **Backend:** Python, Flask
-   **Machine Learning:** Scikit-learn, XGBoost
-   **Data Processing:** Pandas, NumPy
-   **Visualization:** Matplotlib / Plotly
-   **Report Generation:** ReportLab
-   **Frontend:** HTML, CSS, JavaScript
-   **Model Storage:** Pickle (.sav)
-   **Database:** (Future scope - PostgreSQL/MySQL)
-   **Cloud (Future Scope):** AWS / GCP

------------------------------------------------------------------------

## 🔄 Machine Learning Pipeline

### 1️⃣ Data Preprocessing

-   Handling missing values
-   Feature selection
-   Label encoding of categorical variables
-   Normalization / Scaling (if applicable)

### 2️⃣ Model Training

-   Supervised classification model
-   Accuracy achieved: **83.6%**
-   Performance evaluated using accuracy metrics
-   Model exported as `Model.sav`

### 3️⃣ Model Deployment

-   Flask REST API loads serialized model
-   JSON-based input handling
-   Real-time prediction responses

------------------------------------------------------------------------

## 🚀 API Endpoints

### `/predict`

-   Accepts customer features (JSON)
-   Returns:
    -   Churn Prediction (Yes/No)
    -   Probability Score
    -   Risk Level
    -   Confidence Percentage
    -   Risk Factors

### `/analytics`

-   Provides:
    -   Total Customers
    -   Churn Rate
    -   Contract Distribution
    -   Tenure-based churn insights

### `/generate-pdf`

-   Generates executive-level churn analysis report
-   Includes:
    -   Customer profile
    -   Risk assessment
    -   Retention strategy recommendations

------------------------------------------------------------------------

## 📊 Business Intelligence Layer

RetainX combines: - Model-based churn probability - Rule-based risk
interpretation - Automated reporting for business stakeholders

This ensures explainability and executive readiness.

------------------------------------------------------------------------

## 🔍 Key Features

-   End-to-end ML lifecycle implementation
-   Production-ready REST API
-   Automated PDF report generation
-   Business-aligned risk factor analysis
-   Modular system design
-   Scalable architecture (extendable)

------------------------------------------------------------------------

## ⚠️ Identified Improvements (Future Enhancements)

-   Replace inference-time `fit_transform()` with saved encoders
-   Add SHAP explainability
-   Integrate MLflow for model tracking
-   Dockerize for containerized deployment
-   Add authentication and rate limiting
-   Integrate relational database for logging and persistence
-   CI/CD pipeline integration

------------------------------------------------------------------------

## 📈 Resume Value & Impact

RetainX demonstrates: - Applied Machine Learning Engineering - Backend
API development - Data preprocessing and feature engineering -
Production deployment mindset - Business insight generation - Automation
& reporting capabilities

------------------------------------------------------------------------

## 🏁 Conclusion

RetainX represents a scalable, production-oriented Machine Learning
system that bridges predictive modeling with actionable business
intelligence. The architecture supports future enhancements toward MLOps
maturity and cloud-native deployment.

------------------------------------------------------------------------

**Author:** Onkar Shahapurkar\
B.Tech Information Technology (2026 Batch)
