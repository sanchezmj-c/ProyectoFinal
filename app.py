import os
import json

import streamlit as st
import pandas as pd
import numpy as np

from azure.cosmos import CosmosClient
import joblib

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_recall_fscore_support,
    confusion_matrix
)

# -----------------------------
# Config: env vars for secrets
# -----------------------------

COSMOS_URL = os.environ.get("COSMOS_URL", "https://cosmos-mongo-project.documents.azure.com:443/")
COSMOS_KEY = os.environ.get("COSMOS_KEY", "YnVluPqn9xVfyQaQjHPjOQ0N8kl1Vh8fs6mplQCaXM5z7HLBeaLWTHeWydSWlNsNl2C37Ag4in3kACDbc6C3TA==")

DB_NAME = "sample_supplies_db"
CONTAINER_NAME = "sales"

MODEL_PATH = "models/best_coupon_pipeline.pkl"
THRESHOLD_PATH = "models/best_coupon_threshold.json"
RESULTS_PATH = "models/model_results.json"


# -----------------------------
# Helper functions
# -----------------------------

@st.cache_resource(show_spinner=True)
def load_cosmos_client():
    if not COSMOS_URL or not COSMOS_KEY:
        raise RuntimeError("COSMOS_URL or COSMOS_KEY environment variables are not set.")
    client = CosmosClient(COSMOS_URL, credential=COSMOS_KEY)
    return client

@st.cache_data(show_spinner=True)
def load_sales_data():
    client = load_cosmos_client()
    db = client.get_database_client(DB_NAME)
    container = db.get_container_client(CONTAINER_NAME)
    items = list(container.read_all_items())
    df = pd.DataFrame(items)
    return df

def total_amount(items):
    if not isinstance(items, list):
        return np.nan
    return sum((it.get("price", 0) * it.get("quantity", 1)) for it in items)

def total_quantity(items):
    if not isinstance(items, list):
        return np.nan
    return sum(it.get("quantity", 1) for it in items)

def n_unique_items(items):
    if not isinstance(items, list):
        return np.nan
    names = [it.get("name") for it in items if isinstance(it, dict) and "name" in it]
    return len(set(names))

def get_customer_field(c, field):
    if not isinstance(c, dict):
        return np.nan
    return c.get(field)

def build_feature_table(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    if "couponUsed" not in df.columns:
        df["couponUsed"] = np.nan

    if "saleDate" not in df.columns:
        raise KeyError("'saleDate' must be present in df_raw.")

    df["saleDate"] = pd.to_datetime(df["saleDate"], errors="coerce")

    df["total_amount"] = df["items"].apply(total_amount)
    df["n_items"] = df["items"].apply(total_quantity)
    df["n_unique_items"] = df["items"].apply(n_unique_items)

    df["cust_age"] = df["customer"].apply(lambda c: get_customer_field(c, "age"))
    df["cust_gender"] = df["customer"].apply(lambda c: get_customer_field(c, "gender"))
    df["cust_satisfaction"] = df["customer"].apply(lambda c: get_customer_field(c, "satisfaction"))

    df["sale_year"] = df["saleDate"].dt.year
    df["sale_month"] = df["saleDate"].dt.month
    df["sale_dayofweek"] = df["saleDate"].dt.dayofweek
    df["sale_hour"] = df["saleDate"].dt.hour

    feature_cols = [
        "total_amount", "n_items", "n_unique_items",
        "cust_age", "cust_satisfaction",
        "storeLocation", "purchaseMethod", "cust_gender",
        "sale_year", "sale_month", "sale_dayofweek", "sale_hour",
        "couponUsed"
    ]

    feature_cols = [c for c in feature_cols if c in df.columns]
    df_model = df[feature_cols].copy()
    return df_model

@st.cache_resource(show_spinner=True)
def load_model_and_threshold():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Make sure you committed it to GitHub.")
    if not os.path.exists(THRESHOLD_PATH):
        raise FileNotFoundError(f"Threshold file not found at {THRESHOLD_PATH}.")
    pipeline = joblib.load(MODEL_PATH)
    with open(THRESHOLD_PATH, "r") as f:
        data = json.load(f)
    best_threshold = float(data.get("best_threshold", 0.5))
    return pipeline, best_threshold

@st.cache_data(show_spinner=True)
def load_model_results():
    if not os.path.exists(RESULTS_PATH):
        return None
    with open(RESULTS_PATH, "r") as f:
        return json.load(f)

def compute_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = np.nan
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    return acc, auc, prec, rec, f1, cm


# -----------------------------
# Layout
# -----------------------------

st.set_page_config(page_title="Coupon Usage – Cosmos DB + ML Pipelines", layout="wide")

st.title("Coupon Usage Prediction – Cosmos DB + ML Pipelines")

st.markdown(
    """
    This dashboard is part of **Project 2 (Parts 2 & 3)**.  
    It demonstrates an end-to-end pipeline:
    - Data ingestion from **Azure Cosmos DB (NoSQL)**
    - Feature engineering
    - Handling of **class imbalance**
    - Comparison of multiple models (including a Keras MLP trained locally)
    - Deployment of a **scikit-learn pipeline** for real-time inference
    """
)

tab_data, tab_imbalance, tab_models, tab_inference = st.tabs(
    ["Data & EDA", "Class Imbalance", "Models & Results", "Deployed Model Inference"]
)

# Tab 1 – Data
with tab_data:
    st.subheader("1. Data from Cosmos DB")
    try:
        df_sales = load_sales_data()
        st.success(f"Loaded {len(df_sales)} documents from Cosmos DB.")
        st.dataframe(df_sales.head(), use_container_width=True)
    except Exception as e:
        st.error(f"Error loading data from Cosmos DB: {e}")
        st.stop()

    st.markdown(
        """
        The dataset corresponds to the **sample supplies** data in Cosmos DB (`sample_supplies_db.sales`).  
        Each document represents a sale: items, customer info, store, and coupon usage.
        """
    )

# Tab 2 – Imbalance
with tab_imbalance:
    st.subheader("2. Target Variable – Class Imbalance")
    df_sales = load_sales_data()

    if "couponUsed" in df_sales.columns:
        vc = df_sales["couponUsed"].value_counts(dropna=False)
        st.write("Raw `couponUsed` value counts:")
        st.write(vc)
        st.bar_chart(vc)
        st.markdown(
            """
            We observe an **imbalanced classification problem** (~90% no-coupon, ~10% coupon).  
            In Part 2 this was handled by:
            - `class_weight="balanced"` in Logistic Regression and Random Forest  
            - **Threshold tuning** on the validation set to maximize F1 for the minority class
            """
        )
    else:
        st.warning("Column `couponUsed` not found; cannot show imbalance properly.")

# Tab 3 – Models & results
with tab_models:
    st.subheader("3. Models Used and Results (Baseline vs Improved)")
    results = load_model_results()
    if results is None:
        st.error("model_results.json not found. Commit the models/ folder from the notebook.")
    else:
        baseline = pd.DataFrame(results.get("baseline", []))
        improved = results.get("improved_best_sklearn", None)

        st.markdown("### 3.1 Baseline comparison (threshold = 0.5)")
        st.write(
            """
            Validation results with **default threshold 0.5** for:
            - Logistic Regression
            - Random Forest
            - MLP (sklearn)
            - MLP (Keras – trained locally, not deployed)

            Includes **training time** and qualitative **resource requirement**.
            """
        )
        st.dataframe(baseline, use_container_width=True)

        st.markdown("### 3.2 Best scikit-learn model – before vs after threshold tuning")
        if improved is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Baseline (threshold = 0.5)**")
                st.json(improved["baseline"])
            with col2:
                st.markdown("**Improved (tuned threshold)**")
                st.json(improved["improved"])

            st.markdown(
                f"""
                Deployed model: **{improved['model']}** (scikit-learn).  
                Trained with `class_weight="balanced"` and using a tuned decision threshold
                to improve recall/F1 for the minority class.
                """
            )
        else:
            st.info("Improved metrics for best sklearn model not found in model_results.json.")

# Tab 4 – Inference
with tab_inference:
    st.subheader("4. Inference with Deployed scikit-learn Pipeline")

    try:
        pipeline, best_threshold_file = load_model_and_threshold()
        st.success(f"Loaded pipeline from `{MODEL_PATH}` and threshold={best_threshold_file:.2f}")
    except Exception as e:
        st.error(f"Error loading model or threshold: {e}")
        st.stop()

    df_sales = load_sales_data()
    df_model = build_feature_table(df_sales)

    st.write("Feature table preview (after transformation):")
    st.dataframe(df_model.head(), use_container_width=True)

    if "couponUsed" in df_model.columns:
        df_valid = df_model.dropna(subset=["couponUsed"]).copy()
        X_all = df_valid.drop(columns=["couponUsed"])
        y_all = df_valid["couponUsed"].astype(int).values

        y_prob_all = pipeline.predict_proba(X_all)[:, 1]

        thr = st.slider(
            "Decision threshold for classifying coupon usage",
            min_value=0.1,
            max_value=0.9,
            value=float(best_threshold_file),
            step=0.05,
        )

        acc, auc, prec, rec, f1, cm = compute_metrics(y_all, y_prob_all, thr)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{acc:.3f}")
            st.metric("ROC-AUC", f"{auc:.3f}" if not np.isnan(auc) else "N/A")
        with col2:
            st.metric("Precision (class 1)", f"{prec:.3f}")
            st.metric("Recall (class 1)", f"{rec:.3f}")
        with col3:
            st.metric("F1-score (class 1)", f"{f1:.3f}")

        st.write("Confusion matrix (rows=true, cols=predicted):")
        st.write(cm)

        st.markdown(
            """
            **Deployment summary:**
            - All heavy training and threshold tuning was done offline in a Jupyter notebook.
            - Only the final sklearn pipeline and tuned threshold are used here for inference.
            - This keeps the Streamlit app lightweight and runnable on Streamlit Cloud.
            """
        )
    else:
        st.info("No true `couponUsed` labels available; only showing predicted probabilities.")
        X_all = df_model.copy()
        y_prob_all = pipeline.predict_proba(X_all)[:, 1]
        st.write("Sample predicted probabilities:")
        st.write(y_prob_all[:20])
