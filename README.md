
# 🛒 ShopSense Analytics: End-to-End Customer Intelligence Pipeline

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.103.1-009688.svg)
![MLflow](https://img.shields.io/badge/MLflow-Tracking & Registry-blue.svg)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-336791.svg)

## 📌 Project Overview
ShopSense Analytics is a production-ready, end-to-end Data Science pipeline designed to solve three critical business challenges in the e-commerce domain: **Customer Churn Prediction**, **Customer Segmentation**, and **Revenue Forecasting**. 

Unlike standard static modeling projects, this repository simulates a complete real-world Machine Learning lifecycle. It spans from synthetic data engineering and database management to model training, SHAP explainability, MLOps tracking, and finally, a containerized REST API deployment.

---

## 🏗️ Architecture & Pipeline Steps

This project is divided into 5 distinct phases, ensuring a modular and scalable architecture.

### Phase 1: Data Engineering & Exploratory Data Analysis (EDA)
* **Synthetic Data Generation:** Generates realistic e-commerce datasets (Customers, Transactions, Events) featuring log-normal pricing, seasonal trends, and correlated churn triggers.
* **Database Integration:** Automatically provisions and populates a local PostgreSQL data warehouse using SQLAlchemy.
* **Analysis:** Computes univariate statistics, cohort retention tables, and isolates Q4 revenue seasonality.

### Phase 2: Feature Engineering & Hypothesis Testing
* **RFM Analysis:** Calculates Recency, Frequency, and Monetary scores, mapping customers into distinct quintile-based segments.
* **Behavioral Trends:** Uses 12-week linear regression slopes to calculate event recency trends and cart-to-purchase ratios.
* **Statistical Rigor:** Validates business assumptions using non-parametric tests (Mann-Whitney U, Kruskal-Wallis, and Chi-Square).

### Phase 3: Machine Learning Model Development
* **Churn Prediction (Classification):** A scikit-learn pipeline utilizing **XGBoost** to predict customer churn, handling severe class imbalances with dynamic weighting and threshold optimization.
* **Customer Segmentation (Clustering):** Implements **K-Means Clustering** to identify high-value retained customers vs. low-spend flight-risks, validated via Silhouette Scores.
* **Revenue Forecasting (Time Series):** Deploys a **SARIMA** model to predict future monthly revenue, dynamically handling non-stationarity via automated differencing.

### Phase 4: Advanced Explainability & Business Impact
* **SHAP Values:** Integrates TreeExplainer to provide global feature importance and local, individual-level prediction explanations (breaking the "black box").
* **Financial Translation:** Translates ML probabilities into actionable business metrics, calculating the exact **Total Revenue at Risk** and potential ROI of retention campaigns.

### Phase 5: MLOps, Monitoring & Deployment
* **MLflow Tracking & Registry:** Automatically logs model parameters, metrics, and artifacts, registering the best-performing models to a "Staging" environment.
* **Data Drift Detection:** Implements Kolmogorov-Smirnov and Chi-Square tests to monitor feature drift between reference datasets and incoming production data.
* **FastAPI Serving:** Wraps the finalized XGBoost model in a high-performance REST API with dedicated `/health`, `/predict/churn`, and batch inference endpoints.

---

## 🚀 Setup & Installation

### Option A: Fully Containerized Setup (Recommended)
The easiest way to run the entire pipeline—including the Database, MLflow Tracking Server, and FastAPI backend—is via Docker.

1. Ensure Docker and Docker Compose are installed.
2. Clone the repository and navigate to the root directory.
3. Run the following command:
```bash
docker-compose up --build -d

```

4. **Services Started:**
* PostgreSQL Database: `localhost:5432`
* MLflow UI: `http://localhost:5000`
* FastAPI Endpoint: `http://localhost:8000`



### Option B: Local Development Environment

If you wish to run the notebooks or unit tests locally:

1. Create and activate a virtual environment:

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

```

2. Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .

```

3. Run the testing suite:

```bash
pytest tests/

```

---

## 🔌 API Usage Guide

Once the Docker containers or local Uvicorn server is running, you can interact with the API.

**1. Health Check**

```bash
curl http://localhost:8000/health

```

**2. Single Customer Churn Prediction**

```bash
curl -X POST "http://localhost:8000/predict/churn" \
     -H "Content-Type: application/json" \
     -d '{"customer_id": "CUST_123", "customer_features": {"total_sessions": 5, "recency_days": 120, "is_premium": false}}'

```

**Interactive API Documentation:** Navigate to `http://localhost:8000/docs` in your browser to access the Swagger UI.

---

## 👨‍💻 Author

**BILLAKURTI VENKATA SURYANARAYANA** **Roll Number:** 23MH1A4409

**Institution:** Aditya College of Engineering and Technology

*Developed as a comprehensive demonstration of Full-Stack Data Science and Machine Learning Operations.*