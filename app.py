import os
import json

from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np

from azure.cosmos import CosmosClient
import joblib

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_recall_fscore_support,
    confusion_matrix
)

# -------------------------------------------------------------------
# Configuration – environment variables for Cosmos DB + model paths
# -------------------------------------------------------------------

COSMOS_URL = os.environ.get("COSMOS_URL", "https://cosmos-mongo-project.documents.azure.com:443/")
COSMOS_KEY = os.environ.get("COSMOS_KEY", "YnVluPqn9xVfyQaQjHPjOQ0N8kl1Vh8fs6mplQCaXM5z7HLBeaLWTHeWydSWlNsNl2C37Ag4in3kACDbc6C3TA==")

DB_NAME = "sample_supplies_db"
CONTAINER_NAME = "sales"

MODEL_PATH = "models/best_coupon_pipeline.pkl"
THRESHOLD_PATH = "models/best_coupon_threshold.json"
RESULTS_PATH = "models/model_results.json"


# -------------------------------------------------------------------
# Helper functions – data access, feature engineering, models
# -------------------------------------------------------------------

@st.cache_resource(show_spinner=True)
def load_cosmos_client():
    if not COSMOS_URL or not COSMOS_KEY:
        raise RuntimeError("COSMOS_URL or COSMOS_KEY environment variables are not set.")
    client = CosmosClient(COSMOS_URL, credential=COSMOS_KEY)
    return client


@st.cache_data(show_spinner=True)
def load_sales_data():
    """
    Load all sales documents from Cosmos DB into a DataFrame.
    """
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
    """
    Transform raw Cosmos 'sales' documents into the flat feature table
    used to train the models.

    Features:
      - total_amount, n_items, n_unique_items
      - cust_age, cust_satisfaction, cust_gender
      - storeLocation, purchaseMethod
      - sale_year, sale_month, sale_dayofweek, sale_hour

    Target (when available):
      - couponUsed (0/1)
    """
    df = df_raw.copy()

    if "couponUsed" not in df.columns:
        df["couponUsed"] = np.nan

    if "saleDate" not in df.columns:
        raise KeyError("'saleDate' must be present in df_raw.")

    df["saleDate"] = pd.to_datetime(df["saleDate"], errors="coerce")

    # Items / basket features
    df["total_amount"] = df["items"].apply(total_amount)
    df["n_items"] = df["items"].apply(total_quantity)
    df["n_unique_items"] = df["items"].apply(n_unique_items)

    # Customer features
    df["cust_age"] = df["customer"].apply(lambda c: get_customer_field(c, "age"))
    df["cust_gender"] = df["customer"].apply(lambda c: get_customer_field(c, "gender"))
    df["cust_satisfaction"] = df["customer"].apply(lambda c: get_customer_field(c, "satisfaction"))

    # Calendar / time features
    df["sale_year"] = df["saleDate"].dt.year
    df["sale_month"] = df["saleDate"].dt.month
    df["sale_dayofweek"] = df["saleDate"].dt.dayofweek
    df["sale_hour"] = df["saleDate"].dt.hour

    feature_cols = [
        "total_amount", "n_items", "n_unique_items",
        "cust_age", "cust_satisfaction",
        "storeLocation", "purchaseMethod", "cust_gender",
        "sale_year", "sale_month", "sale_dayofweek", "sale_hour",
        "couponUsed",
    ]

    feature_cols = [c for c in feature_cols if c in df.columns]
    df_model = df[feature_cols].copy()
    return df_model


@st.cache_resource(show_spinner=True)
def load_model_and_threshold():
    """
    Load the trained scikit-learn pipeline and tuned decision threshold.
    """
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
    """
    Load model comparison results saved from the training notebook.
    Includes Logistic Regression, Random Forest, MLP (sklearn), and MLP (Keras).
    """
    if not os.path.exists(RESULTS_PATH):
        return None
    with open(RESULTS_PATH, "r") as f:
        return json.load(f)


def compute_metrics(y_true, y_prob, threshold):
    """
    Compute metrics at a given decision threshold for the positive class.
    """
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


@st.cache_data(show_spinner=True)
def prepare_labeled_data_for_pipeline():
    """
    Prepare clean, labeled data for evaluating the deployed pipeline.

    Returns:
        X_all: feature matrix (no NaNs/inf)
        y_all: labels (couponUsed)
        n_total: total rows in df_model
        n_valid_feat: rows that survived cleaning
    """
    df_sales = load_sales_data()
    df_model = build_feature_table(df_sales)

    # Clean infinities -> NaN
    df_model.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop rows with NaNs in feature columns (keep couponUsed for now)
    feature_cols = [c for c in df_model.columns if c != "couponUsed"]
    df_valid = df_model.dropna(subset=feature_cols).copy()

    if df_valid.empty or "couponUsed" not in df_valid.columns:
        return None, None, len(df_model), len(df_valid)

    # Only labeled rows for evaluation
    df_labeled = df_valid.dropna(subset=["couponUsed"]).copy()
    if df_labeled.empty:
        return None, None, len(df_model), len(df_valid)

    X_all = df_labeled.drop(columns=["couponUsed"])
    y_all = df_labeled["couponUsed"].astype(int).values

    n_total = len(df_model)
    n_valid_feat = len(df_valid)
    return X_all, y_all, n_total, n_valid_feat


# -------------------------------------------------------------------
# Streamlit layout & pages
# -------------------------------------------------------------------

st.set_page_config(
    page_title="Coupon Usage Prediction – Cosmos DB & Machine Learning",
    layout="wide"
)

st.title("Coupon Usage Prediction Dashboard")

st.markdown(
    """
    This dashboard showcases an end-to-end **machine learning solution** built on top of  
    **Azure Cosmos DB (NoSQL)** and Python.

    **Business question:**  
    > Given information about a sale (basket, customer, store, and time),  
    > what is the probability that the customer uses a coupon?

    The app demonstrates:
    - Real-time data access from **Cosmos DB**
    - Feature engineering and handling of **class imbalance**
    - Comparison of multiple models (Logistic Regression, Random Forest, MLP, Keras MLP)
    - Deployment of a **scikit-learn pipeline** for inference
    """
)

tab_data, tab_imbalance, tab_models, tab_inference = st.tabs(
    ["Data Overview", "Target Balance", "Model Performance", "Deployed Model & Predictions"]
)

# -------------------------------------------------------------------
# Tab 1 – Data Overview
# -------------------------------------------------------------------

with tab_data:
    st.subheader("1. Data Overview – Sales in Cosmos DB")

    try:
        df_sales = load_sales_data()
        st.success(f"Connected to Cosmos DB. Retrieved **{len(df_sales)}** sales documents.")
        st.dataframe(df_sales.head(), use_container_width=True)
    except Exception as e:
        st.error(f"Error loading data from Cosmos DB: {e}")
        st.stop()

    st.markdown(
        """
        Each row represents a **sale transaction** with:
        - Items purchased (name, price, quantity)
        - Customer demographics and satisfaction
        - Store location and purchase method
        - The field **`couponUsed`** indicating whether a coupon was applied in that sale.
        """
    )

# -------------------------------------------------------------------
# Tab 2 – Target Balance
# -------------------------------------------------------------------

with tab_imbalance:
    st.subheader("2. Target Balance – How Often Are Coupons Used?")

    df_sales = load_sales_data()

    if "couponUsed" in df_sales.columns:
        vc = df_sales["couponUsed"].value_counts(dropna=False)
        st.write("Distribution of `couponUsed` in the raw data:")
        st.write(vc)

        st.bar_chart(vc)

        st.markdown(
            """
            The chart confirms a **highly imbalanced problem**:
            - The majority class corresponds to **sales without a coupon**
            - The minority class corresponds to **sales where a coupon was used**

            In the modeling phase we addressed this by:
            - Using `class_weight="balanced"` in several models
            - Performing **decision threshold tuning** to increase recall on coupon users
            """
        )
    else:
        st.warning("Column `couponUsed` not found; target balance cannot be displayed.")

# -------------------------------------------------------------------
# Tab 3 – Model Performance (with confusion matrices)
# -------------------------------------------------------------------

with tab_models:
    st.subheader("3. Model Performance – Training & Validation Results")

    results = load_model_results()
    if results is None:
        st.error("`model_results.json` not found. Please commit the `models/` folder from the training notebook.")
    else:
        baseline = pd.DataFrame(results.get("baseline", []))
        improved = results.get("improved_best_sklearn", None)

        st.markdown("### 3.1 Comparison of Models (Validation Set, Threshold = 0.5)")
        st.write(
            """
            The table below summarizes the **validation performance** of all models
            using the **default decision threshold 0.5**:
            - Logistic Regression
            - Random Forest
            - MLP (scikit-learn)
            - MLP (Keras, trained locally on GPU – not deployed)
            """
        )
        st.dataframe(baseline, use_container_width=True)

        st.markdown("### 3.2 Best scikit-learn Model – Baseline vs Tuned Threshold")

        if improved is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Baseline configuration (threshold = 0.5)**")
                st.json(improved["baseline"])
            with col2:
                st.markdown("**Improved configuration (tuned threshold)**")
                st.json(improved["improved"])

            st.markdown(
                f"""
                The **deployed model** is: **{improved['model']}** (scikit-learn).  
                It is trained with `class_weight="balanced"` and uses a **custom decision threshold**
                to improve detection of coupon users.
                """
            )

            # --- Confusion matrices computed on current labeled data ---
            st.markdown("### 3.3 Confusion Matrices for the Deployed Model")

            try:
                pipeline, best_threshold = load_model_and_threshold()
                X_all, y_all, n_total, n_valid_feat = prepare_labeled_data_for_pipeline()

                if X_all is None:
                    st.info("No labeled rows with valid features available to compute confusion matrices.")
                else:
                    y_prob_all = pipeline.predict_proba(X_all)[:, 1]

                    # Baseline threshold 0.5
                    acc_05, auc_05, prec_05, rec_05, f1_05, cm_05 = compute_metrics(
                        y_all, y_prob_all, 0.5
                    )
                    # Tuned threshold from training
                    acc_t, auc_t, prec_t, rec_t, f1_t, cm_t = compute_metrics(
                        y_all, y_prob_all, best_threshold
                    )

                    colA, colB = st.columns(2)
                    with colA:
                        st.markdown("**Baseline threshold = 0.5**")
                        df_cm_05 = pd.DataFrame(
                            cm_05,
                            index=["True 0 (no coupon)", "True 1 (coupon)"],
                            columns=["Pred 0", "Pred 1"],
                        )
                        st.table(df_cm_05)

                    with colB:
                        st.markdown(f"**Tuned threshold = {best_threshold:.2f}**")
                        df_cm_t = pd.DataFrame(
                            cm_t,
                            index=["True 0 (no coupon)", "True 1 (coupon)"],
                            columns=["Pred 0", "Pred 1"],
                        )
                        st.table(df_cm_t)

                    st.caption(
                        f"Confusion matrices computed on {len(y_all)} labeled rows "
                        f"with valid features (out of {n_total} total sales; "
                        f"{n_total - n_valid_feat} rows were dropped due to missing or invalid values)."
                    )

            except Exception as e:
                st.error(f"Error computing confusion matrices for the deployed model: {e}")
        else:
            st.info("Improved metrics for the best scikit-learn model were not found in `model_results.json`.")

# -------------------------------------------------------------------
# Tab 4 – Deployed Model & Predictions
# -------------------------------------------------------------------

with tab_inference:
    st.subheader("4. Deployed Model & Interactive Predictions")

    st.markdown(
        """
        The deployed model is a **scikit-learn pipeline** that predicts:

        > The probability that a given sale will use a coupon  
        > (i.e., `P(couponUsed = 1 | transaction information)`)

        This section has two parts:
        - Batch evaluation on historical data
        - A **“what-if” simulator** for new scenarios
        """
    )

    try:
        pipeline, best_threshold = load_model_and_threshold()
        st.success(
            f"Loaded trained pipeline from `{MODEL_PATH}` "
            f"and tuned threshold = {best_threshold:.2f}"
        )
    except Exception as e:
        st.error(f"Error loading model or threshold: {e}")
        st.stop()

    # --- Part A: batch evaluation with interactive threshold ---

    st.markdown("### 4.1 Performance on Historical Labeled Data")

    X_all, y_all, n_total, n_valid_feat = prepare_labeled_data_for_pipeline()

    if X_all is None:
        st.info(
            "There are no sufficiently clean, labeled rows to evaluate the deployed model. "
            "Check data quality in Cosmos DB."
        )
    else:
        y_prob_all = pipeline.predict_proba(X_all)[:, 1]

        thr = st.slider(
            "Decision threshold for classifying coupon usage",
            min_value=0.1,
            max_value=0.9,
            value=float(best_threshold),
            step=0.05,
        )

        acc, auc, prec, rec, f1, cm = compute_metrics(y_all, y_prob_all, thr)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{acc:.3f}")
            st.metric("ROC-AUC", f"{auc:.3f}" if not np.isnan(auc) else "N/A")
        with col2:
            st.metric("Precision (coupon = 1)", f"{prec:.3f}")
            st.metric("Recall (coupon = 1)", f"{rec:.3f}")
        with col3:
            st.metric("F1-score (coupon = 1)", f"{f1:.3f}")

        st.write("Confusion matrix (rows = true class, columns = predicted class):")
        df_cm = pd.DataFrame(
            cm,
            index=["True 0 (no coupon)", "True 1 (coupon)"],
            columns=["Pred 0", "Pred 1"],
        )
        st.table(df_cm)

        st.caption(
            f"Evaluation based on {len(y_all)} labeled rows with valid features. "
            f"{n_total - n_valid_feat} rows were excluded due to missing or invalid values."
        )

    # --- Part B: manual “what-if” prediction form ---

    st.markdown("### 4.2 What-If Scenario: Predict Coupon Usage for a Single Sale")

    # Build df_model to derive realistic ranges and categories
    df_sales = load_sales_data()
    df_model = build_feature_table(df_sales)
    df_model.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Categorical options from actual data (with fallbacks)
    store_locations = sorted(
        df_model["storeLocation"].dropna().unique().tolist()
    ) if "storeLocation" in df_model.columns else []
    purchase_methods = sorted(
        df_model["purchaseMethod"].dropna().unique().tolist()
    ) if "purchaseMethod" in df_model.columns else []
    genders = sorted(
        df_model["cust_gender"].dropna().unique().tolist()
    ) if "cust_gender" in df_model.columns else []

    if not store_locations:
        store_locations = ["Denver", "Seattle", "London", "Austin", "New York", "San Diego"]
    if not purchase_methods:
        purchase_methods = ["Online", "Phone", "In store"]
    if not genders:
        genders = ["M", "F"]

    # Numeric ranges based on actual data
    def num_range(col_name, default_min, default_max, default_val):
        if col_name in df_model.columns:
            series = df_model[col_name].dropna()
            if not series.empty:
                return float(series.min()), float(series.max()), float(series.median())
        return default_min, default_max, default_val

    amt_min, amt_max, amt_med = num_range("total_amount", 0.0, 10000.0, 1000.0)
    items_min, items_max, items_med = num_range("n_items", 1.0, 100.0, 20.0)
    uniq_min, uniq_max, uniq_med = num_range("n_unique_items", 1.0, 10.0, 4.0)
    age_min, age_max, age_med = num_range("cust_age", 18.0, 80.0, 40.0)
    sat_min, sat_max, sat_med = num_range("cust_satisfaction", 1.0, 5.0, 4.0)

    # Modes for time features
    def mode_or_default(col_name, default):
        if col_name in df_model.columns:
            series = df_model[col_name].dropna()
            if not series.empty:
                return int(series.mode()[0])
        return default

    default_month = mode_or_default("sale_month", 6)
    default_dow = mode_or_default("sale_dayofweek", 3)
    default_hour = mode_or_default("sale_hour", 12)
    default_year = mode_or_default("sale_year", 2015)

    with st.form("manual_prediction_form"):
        st.markdown("**Customer profile**")
        cust_age = st.number_input(
            "Customer age",
            min_value=int(age_min),
            max_value=int(age_max),
            value=int(age_med),
        )
        cust_satisfaction = st.slider(
            "Customer satisfaction (1–5)",
            min_value=int(sat_min),
            max_value=int(sat_max),
            value=int(sat_med),
        )
        cust_gender = st.selectbox("Customer gender", genders)

        st.markdown("**Transaction details**")
        total_amount = st.number_input(
            "Total basket amount",
            min_value=float(amt_min),
            max_value=float(amt_max),
            value=float(amt_med),
            step=10.0,
        )
        n_items = st.number_input(
            "Total quantity of items",
            min_value=int(items_min),
            max_value=int(items_max),
            value=int(items_med),
            step=1,
        )
        n_unique_items = st.number_input(
            "Number of different items",
            min_value=int(uniq_min),
            max_value=int(uniq_max),
            value=int(uniq_med),
            step=1,
        )

        st.markdown("**Store & channel**")
        store_location = st.selectbox("Store location", store_locations)
        purchase_method = st.selectbox("Purchase method", purchase_methods)

        st.markdown("**Time of sale**")
        sale_month = st.slider(
            "Month of sale (1 = Jan, 12 = Dec)",
            min_value=1,
            max_value=12,
            value=int(default_month),
        )
        sale_dayofweek = st.slider(
            "Day of week (0 = Monday, 6 = Sunday)",
            min_value=0,
            max_value=6,
            value=int(default_dow),
        )
        sale_hour = st.slider(
            "Hour of day (0–23)",
            min_value=0,
            max_value=23,
            value=int(default_hour),
        )

        submitted = st.form_submit_button("Estimate coupon usage probability")

    if submitted:
        try:
            # Construct a single-row DataFrame with the exact same feature names as training
            single_row = pd.DataFrame([{
                "total_amount": float(total_amount),
                "n_items": int(n_items),
                "n_unique_items": int(n_unique_items),
                "cust_age": float(cust_age),
                "cust_satisfaction": int(cust_satisfaction),
                "storeLocation": store_location,
                "purchaseMethod": purchase_method,
                "cust_gender": cust_gender,
                "sale_year": int(default_year),   # any realistic year in the training range
                "sale_month": int(sale_month),
                "sale_dayofweek": int(sale_dayofweek),
                "sale_hour": int(sale_hour),
            }])

            prob_coupon = pipeline.predict_proba(single_row)[:, 1][0]
            pred_label = int(prob_coupon >= best_threshold)

            st.markdown("#### Prediction result")
            st.write(f"**Estimated probability of coupon usage:** `{prob_coupon:.3f}`")
            st.write(f"**Decision threshold in use:** `{best_threshold:.2f}`")

            if pred_label == 1:
                st.success("Model decision: **Coupon usage likely (class 1).**")
            else:
                st.info("Model decision: **Coupon usage unlikely (class 0).**")

            st.caption(
                "This prediction is made by the same pipeline used in model evaluation. "
                "Only inference is performed in the app; training and tuning were done offline."
            )
        except Exception as e:
            st.error(f"Error computing manual prediction: {e}")
