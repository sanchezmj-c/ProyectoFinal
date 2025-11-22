import os
import json

import streamlit as st
import pandas as pd
import numpy as np

from azure.cosmos import CosmosClient
import joblib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

# -------------------------------------------------------------------
# Configuration – environment variables & model paths
# -------------------------------------------------------------------

COSMOS_URL = os.environ.get("COSMOS_URL", "https://cosmos-mongo-project.documents.azure.com:443/")
COSMOS_KEY = os.environ.get("COSMOS_KEY", "YnVluPqn9xVfyQaQjHPjOQ0N8kl1Vh8fs6mplQCaXM5z7HLBeaLWTHeWydSWlNsNl2C37Ag4in3kACDbc6C3TA==")

DB_NAME = "sample_supplies_db"
CONTAINER_NAME = "sales"

MODEL_PATH = "models/best_coupon_pipeline.pkl"
THRESHOLD_PATH = "models/best_coupon_threshold.json"
RESULTS_PATH = "models/model_results.json"

TARGET_COL = "couponUsed"

# Final selected features used in retraining notebook
SELECTED_FEATURES = [
    "total_amount",
    "n_items",
    "n_unique_items",
    "cust_age",
    "cust_gender",
    "storeLocation",
    "purchaseMethod",
    "sale_month",
]


# -------------------------------------------------------------------
# Data access & feature engineering
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
    used by the FINAL (retrained) model.

    Features:
      - total_amount, n_items, n_unique_items
      - cust_age, cust_gender
      - storeLocation, purchaseMethod
      - sale_month

    Target:
      - couponUsed (0/1)
    """
    df = df_raw.copy()

    if TARGET_COL not in df.columns:
        raise KeyError(f"'{TARGET_COL}' must be present in df_raw.")
    if "saleDate" not in df.columns:
        raise KeyError("'saleDate' must be present in df_raw.")

    # Keep only rows with known target
    df = df[df[TARGET_COL].notna()].copy()
    df[TARGET_COL] = df[TARGET_COL].astype(bool)

    # Parse date
    df["saleDate"] = pd.to_datetime(df["saleDate"], errors="coerce")

    # Basket features
    df["total_amount"] = df["items"].apply(total_amount)
    df["n_items"] = df["items"].apply(total_quantity)
    df["n_unique_items"] = df["items"].apply(n_unique_items)

    # Customer features
    df["cust_age"] = df["customer"].apply(lambda c: get_customer_field(c, "age"))
    df["cust_gender"] = df["customer"].apply(lambda c: get_customer_field(c, "gender"))

    # Time features (only month kept in final model)
    df["sale_month"] = df["saleDate"].dt.month

    cols_needed = SELECTED_FEATURES + [TARGET_COL]
    df_model = df[cols_needed].copy()

    # Clean infinities and drop rows with any NaNs in features or target
    df_model.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_model = df_model.dropna(subset=cols_needed)

    df_model[TARGET_COL] = df_model[TARGET_COL].astype(int)
    return df_model


# -------------------------------------------------------------------
# Model loading & metrics helpers
# -------------------------------------------------------------------

@st.cache_resource(show_spinner=True)
def load_model_and_threshold():
    """
    Load the retrained best sklearn pipeline (reduced features)
    and the tuned decision threshold.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}.")
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
    Load model_results.json with metrics for:
      - Logistic Regression
      - Random Forest (lighter)
      - MLP (sklearn)
      - LightGBM
      - MLP (Keras)
    All trained on the reduced feature set, each evaluated at its own
    best validation threshold (F1-optimized).
    """
    if not os.path.exists(RESULTS_PATH):
        return None
    with open(RESULTS_PATH, "r") as f:
        return json.load(f)


def compute_metrics(y_true, y_prob, threshold):
    """
    Compute metrics at a given decision threshold.
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
    Prepare clean, labeled data for evaluating the deployed pipeline
    on CURRENT Cosmos DB data.

    Returns:
        X_all: feature matrix (SELECTED_FEATURES)
        y_all: labels (couponUsed)
        n_total: total rows in feature table
        n_valid_feat: rows kept after cleaning
    """
    df_sales = load_sales_data()
    df_model = build_feature_table(df_sales)  # already cleans and selects features

    X_all = df_model[SELECTED_FEATURES].copy()
    y_all = df_model[TARGET_COL].values

    n_total = len(df_model)
    n_valid_feat = len(df_model)  # all kept, build_feature_table already dropped bad rows

    return X_all, y_all, n_total, n_valid_feat


# -------------------------------------------------------------------
# Streamlit page configuration
# -------------------------------------------------------------------

st.set_page_config(
    page_title="Coupon Usage Prediction – Cosmos DB & Machine Learning",
    layout="wide",
)

st.title("Coupon Usage Prediction Dashboard")

st.markdown(
    """
    This dashboard showcases an end-to-end **machine learning solution** built on:
    - **Azure Cosmos DB (NoSQL)** for operational sales data  
    - **Python, scikit-learn, LightGBM & Keras** for modeling  
    - **Streamlit** for interactive reporting and inference  

    **Business question:**  
    > Given information about a sale (basket, customer, store, and timing),  
    > what is the probability that the customer uses a coupon?

    The app demonstrates:
    - Real-time data access from Cosmos DB  
    - Handling an **imbalanced classification problem**  
    - Comparison of 5 models  
    - Deployment of the **best sklearn model** on a compact feature set  
    """
)

tab_data, tab_imbalance, tab_models, tab_inference = st.tabs(
    ["Data Overview & EDA", "Target Balance", "Model Performance", "Deployed Model & Predictions"]
)

# -------------------------------------------------------------------
# Tab 1 – Data Overview & EDA (with filters & timelines)
# -------------------------------------------------------------------

with tab_data:
    st.subheader("1. Data Overview, Filters & Exploratory Analysis")

    try:
        df_sales = load_sales_data()
        st.success(f"Connected to Cosmos DB. Retrieved **{len(df_sales)}** sales documents.")
        st.markdown("#### Raw sales documents (sample)")
        st.dataframe(df_sales.head(), use_container_width=True)
    except Exception as e:
        st.error(f"Error loading data from Cosmos DB: {e}")
        st.stop()

    st.markdown(
        """
        Each document represents a **sale transaction** with:
        - Items purchased (name, price, quantity)  
        - Embedded customer demographics  
        - Store location and purchase method  
        - The flag **`couponUsed`** indicating whether a coupon was applied.  
        """
    )

    st.markdown("#### Engineered feature table (used for modeling)")

    try:
        df_model = build_feature_table(df_sales)
        st.write(f"Feature table rows after cleaning: **{len(df_model)}**")
    except Exception as e:
        st.error(f"Error building feature table: {e}")
        df_model = None

    if df_model is None or df_model.empty:
        st.warning("Feature table is empty; cannot perform EDA.")
    else:
        # ---------------- Filters ----------------
        st.markdown("##### Filters for EDA (subset of data)")

        # Prepare filter options
        stores_all = (
            sorted(df_model["storeLocation"].dropna().unique().tolist())
            if "storeLocation" in df_model.columns
            else []
        )
        methods_all = (
            sorted(df_model["purchaseMethod"].dropna().unique().tolist())
            if "purchaseMethod" in df_model.columns
            else []
        )
        months_all = (
            sorted(df_model["sale_month"].dropna().unique().tolist())
            if "sale_month" in df_model.columns
            else list(range(1, 13))
        )

        default_month_min = min(months_all) if months_all else 1
        default_month_max = max(months_all) if months_all else 12

        with st.expander("Open filters", expanded=True):
            col_f1, col_f2, col_f3 = st.columns(3)

            with col_f1:
                store_filter = st.multiselect(
                    "Store location",
                    options=stores_all,
                    default=stores_all,
                    help="Filter EDA charts by selected stores.",
                )

            with col_f2:
                method_filter = st.multiselect(
                    "Purchase method",
                    options=methods_all,
                    default=methods_all,
                    help="Filter EDA charts by selected channels.",
                )

            with col_f3:
                month_range = st.slider(
                    "Sale month range",
                    min_value=int(default_month_min),
                    max_value=int(default_month_max),
                    value=(int(default_month_min), int(default_month_max)),
                    step=1,
                    help="Filter EDA charts by sale month.",
                )

        # Apply filters
        df_filt = df_model.copy()

        if store_filter and "storeLocation" in df_filt.columns:
            df_filt = df_filt[df_filt["storeLocation"].isin(store_filter)]
        if method_filter and "purchaseMethod" in df_filt.columns:
            df_filt = df_filt[df_filt["purchaseMethod"].isin(method_filter)]
        if "sale_month" in df_filt.columns and month_range:
            df_filt = df_filt[
                (df_filt["sale_month"] >= month_range[0])
                & (df_filt["sale_month"] <= month_range[1])
            ]

        st.markdown(
            f"Filtered subset: **{len(df_filt)}** rows (out of {len(df_model)} total after cleaning)."
        )
        st.dataframe(df_filt.head(), use_container_width=True)

        if df_filt.empty:
            st.warning("No rows match the selected filters. Adjust filters to see EDA.")
        else:
            # ----- Numeric stats on filtered subset -----
            st.markdown("##### Descriptive statistics (numeric features, filtered subset)")
            num_cols_for_desc = ["total_amount", "n_items", "n_unique_items", "cust_age"]
            num_cols_for_desc = [c for c in num_cols_for_desc if c in df_filt.columns]
            if num_cols_for_desc:
                st.dataframe(df_filt[num_cols_for_desc].describe().T)
            else:
                st.info("No numeric columns available for statistics.")

            colA, colB = st.columns(2)

            with colA:
                st.markdown("##### Sales count by store location (filtered)")
                if "storeLocation" in df_filt.columns:
                    vc_store = (
                        df_filt["storeLocation"]
                        .value_counts()
                        .rename_axis("storeLocation")
                        .to_frame("count")
                    )
                    st.bar_chart(vc_store)
                else:
                    st.info("Column `storeLocation` not found in feature table.")

            with colB:
                st.markdown("##### Sales count by purchase method (filtered)")
                if "purchaseMethod" in df_filt.columns:
                    vc_pm = (
                        df_filt["purchaseMethod"]
                        .value_counts()
                        .rename_axis("purchaseMethod")
                        .to_frame("count")
                    )
                    st.bar_chart(vc_pm)
                else:
                    st.info("Column `purchaseMethod` not found in feature table.")

            st.markdown("##### Average basket size and value by coupon usage (filtered)")

            if TARGET_COL in df_filt.columns:
                agg = (
                    df_filt.groupby(TARGET_COL)
                    .agg(
                        avg_total_amount=("total_amount", "mean"),
                        avg_n_items=("n_items", "mean"),
                        avg_n_unique_items=("n_unique_items", "mean"),
                    )
                    .rename(index={0: "No coupon", 1: "Coupon used"})
                )
                st.dataframe(agg)
                st.caption(
                    "These averages are computed on the filtered subset. "
                    "They help compare basket characteristics between transactions with and without coupons."
                )
            else:
                st.info(f"Column `{TARGET_COL}` not found in feature table.")

            # ----- Timeline of coupon usage by month -----
            st.markdown("##### Timeline of coupon usage by month (filtered subset)")

            if {"sale_month", TARGET_COL}.issubset(df_filt.columns):
                month_agg = (
                    df_filt.groupby("sale_month")[TARGET_COL]
                    .agg(coupon_rate="mean", n_sales="count")
                    .reset_index()
                )
                # Simple line chart of coupon rate
                month_agg_display = month_agg.set_index("sale_month")[["coupon_rate"]]
                st.line_chart(month_agg_display)
                st.caption(
                    "Coupon rate = average of `couponUsed` per month, on the filtered subset."
                )
            else:
                st.info("Cannot compute monthly timeline; missing `sale_month` or target.")

            # ----- Timeline by month and store -----
            st.markdown("##### Coupon usage by month and store (coupon rate)")

            if {"sale_month", "storeLocation", TARGET_COL}.issubset(df_filt.columns):
                # Group by month and store, then pivot to wide format for multi-line chart
                store_month_agg = (
                    df_filt.groupby(["sale_month", "storeLocation"])[TARGET_COL]
                    .mean()
                    .reset_index()
                )
                pivot_store_month = store_month_agg.pivot(
                    index="sale_month",
                    columns="storeLocation",
                    values=TARGET_COL,
                )
                # For readability, limit to a reasonable number of stores
                if pivot_store_month.shape[1] > 10:
                    top_stores = (
                        df_filt.groupby("storeLocation")[TARGET_COL]
                        .count()
                        .sort_values(ascending=False)
                        .head(10)
                        .index
                        .tolist()
                    )
                    pivot_store_month = pivot_store_month[top_stores]

                st.line_chart(pivot_store_month)
                st.caption(
                    "Each line shows the coupon usage rate per month for a store "
                    "in the filtered subset. If many stores exist, only the top few by volume are shown."
                )
            else:
                st.info(
                    "Cannot compute monthly timeline by store; missing `sale_month`, `storeLocation`, or target."
                )

# -------------------------------------------------------------------
# Tab 2 – Target Balance
# -------------------------------------------------------------------

with tab_imbalance:
    st.subheader("2. Target Balance – How Often Are Coupons Used?")

    df_sales = load_sales_data()

    if TARGET_COL in df_sales.columns:
        vc = df_sales[TARGET_COL].value_counts(dropna=False)
        st.write("Distribution of `couponUsed` in the raw data:")
        st.write(vc)

        st.bar_chart(vc)

        st.markdown(
            """
            The dataset is **highly imbalanced**:
            - The majority class corresponds to **sales without a coupon**
            - The minority class corresponds to **sales where a coupon was used**

            During training, this imbalance was addressed via:
            - `class_weight="balanced"` in all sklearn models  
            - Per-model **threshold tuning** on the validation set, focusing on
              better F1-score for the coupon (positive) class.  
            """
        )
    else:
        st.warning(f"Column `{TARGET_COL}` not found; target balance cannot be displayed.")

# -------------------------------------------------------------------
# Tab 3 – Model Performance (5 models)
# -------------------------------------------------------------------

with tab_models:
    st.subheader("3. Model Performance – Validation & Test Results")

    results_json = load_model_results()
    if results_json is None:
        st.error("`model_results.json` not found. Please commit the `models/` folder from the retraining notebook.")
    else:
        baseline = pd.DataFrame(results_json.get("baseline", []))
        improved = results_json.get("improved_best_sklearn", None)

        st.markdown("### 3.1 Comparison of Models (Validation Set, Best Threshold per Model)")
        st.write(
            """
            The table below summarizes **validation-set performance** for the 5 candidate models,
            each evaluated at its **own best threshold** (selected to maximize F1 on the validation set):

            - Logistic Regression (sklearn)  
            - Random Forest (sklearn, lighter configuration)  
            - MLP (sklearn)  
            - LightGBM  
            - MLP (Keras, trained locally with GPU – comparison only)  
            """
        )
        st.dataframe(baseline, use_container_width=True)

        st.markdown("### 3.2 Deployed Model – Best sklearn Model (Reduced Features)")

        if improved is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Validation metrics (at model's best threshold)**")
                st.json(improved["baseline"])
            with col2:
                st.markdown("**Test metrics (same threshold, unseen data)**")
                st.json(improved["improved"])

            st.markdown(
                f"""
                The deployed model is the **best sklearn model on the reduced feature set**:
                **{improved['model']}**.

                For this model:
                - The validation-set threshold was tuned to maximize F1 for coupon usage.  
                - The same threshold was then applied to the test set to measure **generalization**.  

                All models share the same input features:
                - Basket size and value  
                - Customer age and gender  
                - Store location and purchase channel  
                - Month of the sale  
                """
            )
        else:
            st.info("Improved metrics for the best sklearn model were not found in `model_results.json`.")

        # Confusion matrices on current Cosmos DB data (for deployed model)
        st.markdown("### 3.3 Confusion Matrices on Current Cosmos Data (Deployed Model)")

        try:
            pipeline, best_threshold = load_model_and_threshold()
            X_all, y_all, n_total, n_valid_feat = prepare_labeled_data_for_pipeline()

            if X_all is None or len(X_all) == 0:
                st.info("No labeled rows with valid features to compute confusion matrices.")
            else:
                y_prob_all = pipeline.predict_proba(X_all)[:, 1]

                # Baseline threshold 0.5 (reference)
                acc_05, auc_05, prec_05, rec_05, f1_05, cm_05 = compute_metrics(
                    y_all, y_prob_all, 0.5
                )
                # Tuned threshold from training
                acc_t, auc_t, prec_t, rec_t, f1_t, cm_t = compute_metrics(
                    y_all, y_prob_all, best_threshold
                )

                colA, colB = st.columns(2)
                with colA:
                    st.markdown("**Reference threshold = 0.5**")
                    df_cm_05 = pd.DataFrame(
                        cm_05,
                        index=["True 0 (no coupon)", "True 1 (coupon)"],
                        columns=["Pred 0", "Pred 1"],
                    )
                    st.table(df_cm_05)
                    st.caption(
                        f"Accuracy={acc_05:.3f}, Precision={prec_05:.3f}, "
                        f"Recall={rec_05:.3f}, F1={f1_05:.3f}"
                    )

                with colB:
                    st.markdown(f"**Deployed threshold = {best_threshold:.2f}**")
                    df_cm_t = pd.DataFrame(
                        cm_t,
                        index=["True 0 (no coupon)", "True 1 (coupon)"],
                        columns=["Pred 0", "Pred 1"],
                    )
                    st.table(df_cm_t)
                    st.caption(
                        f"Accuracy={acc_t:.3f}, Precision={prec_t:.3f}, "
                        f"Recall={rec_t:.3f}, F1={f1_t:.3f}"
                    )

                st.caption(
                    f"Confusion matrices computed on {len(y_all)} labeled rows with valid features "
                    f"from the current Cosmos DB data."
                )
        except Exception as e:
            st.error(f"Error computing confusion matrices for the deployed model: {e}")

# -------------------------------------------------------------------
# Tab 4 – Deployed Model & Predictions
# -------------------------------------------------------------------

with tab_inference:
    st.subheader("4. Deployed Model & Interactive Predictions")

    st.markdown(
        """
        The deployed model is a sklearn pipeline trained on the following features:
        - `total_amount` – total value of the basket  
        - `n_items` – total quantity purchased  
        - `n_unique_items` – number of distinct products  
        - `cust_age` – customer age  
        - `cust_gender` – customer gender  
        - `storeLocation` – store / city  
        - `purchaseMethod` – channel (in-store, online, phone)  
        - `sale_month` – month of the sale  

        It predicts:

        > **P(couponUsed = 1 | selected features)** – the probability that the sale uses a coupon.
        """
    )

    try:
        pipeline, best_threshold = load_model_and_threshold()
        st.success(
            f"Loaded deployed pipeline from `{MODEL_PATH}` "
            f"with tuned threshold = {best_threshold:.2f}"
        )
    except Exception as e:
        st.error(f"Error loading model or threshold: {e}")
        st.stop()

    # --- Part A: Batch evaluation on current Cosmos DB data ---

    st.markdown("### 4.1 Performance on Current Cosmos DB Data")

    try:
        X_all, y_all, n_total, n_valid_feat = prepare_labeled_data_for_pipeline()
    except Exception as e:
        st.error(f"Error preparing labeled data for pipeline evaluation: {e}")
        X_all, y_all, n_total, n_valid_feat = None, None, 0, 0

    if X_all is None or len(X_all) == 0:
        st.info(
            "There are no sufficiently clean, labeled rows with the selected features "
            "to evaluate the deployed model."
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
            f"Evaluation based on {len(y_all)} labeled rows with valid features from Cosmos DB."
        )

    # --- Part B: Manual “what-if” prediction form ---

    st.markdown("### 4.2 What-If Scenario – Predict Coupon Usage for a Single Sale")

    # Build df_model to derive realistic ranges and categories
    try:
        df_sales = load_sales_data()
        df_model = build_feature_table(df_sales)
    except Exception as e:
        st.error(f"Error rebuilding feature table for manual prediction defaults: {e}")
        df_model = None

    store_locations = []
    purchase_methods = []
    genders = []
    amt_min = amt_max = amt_med = 0.0
    items_min = items_max = items_med = 1
    uniq_min = uniq_max = uniq_med = 1
    age_min = age_max = age_med = 30
    default_month = 6

    if df_model is not None and not df_model.empty:
        if "storeLocation" in df_model.columns:
            store_locations = sorted(df_model["storeLocation"].dropna().unique().tolist())
        if "purchaseMethod" in df_model.columns:
            purchase_methods = sorted(df_model["purchaseMethod"].dropna().unique().tolist())
        if "cust_gender" in df_model.columns:
            genders = sorted(df_model["cust_gender"].dropna().unique().tolist())

        def num_range(col_name, default_min, default_max, default_val):
            if col_name in df_model.columns:
                series = df_model[col_name].dropna()
                if not series.empty:
                    return float(series.min()), float(series.max()), float(series.median())
            return default_min, default_max, default_val

        amt_min, amt_max, amt_med = num_range("total_amount", 0.0, 10000.0, 100.0)
        items_min, items_max, items_med = num_range("n_items", 1.0, 100.0, 5.0)
        uniq_min, uniq_max, uniq_med = num_range("n_unique_items", 1.0, 10.0, 3.0)
        age_min, age_max, age_med = num_range("cust_age", 18.0, 80.0, 40.0)

        if "sale_month" in df_model.columns:
            series_m = df_model["sale_month"].dropna()
            if not series_m.empty:
                default_month = int(series_m.mode()[0])

    # Fallbacks if dataset was empty or missing some values
    if not store_locations:
        store_locations = ["Denver", "Seattle", "London", "Austin", "New York", "San Diego"]
    if not purchase_methods:
        purchase_methods = ["In store", "Online", "Phone"]
    if not genders:
        genders = ["M", "F"]

    with st.form("manual_prediction_form"):
        st.markdown("**Customer profile**")
        cust_age = st.number_input(
            "Customer age",
            min_value=int(age_min),
            max_value=int(age_max),
            value=int(age_med),
        )
        cust_gender = st.selectbox("Customer gender", genders)

        st.markdown("**Basket details**")
        total_amount_val = st.number_input(
            "Total basket amount",
            min_value=float(amt_min),
            max_value=float(amt_max),
            value=float(amt_med),
            step=10.0,
        )
        n_items_val = st.number_input(
            "Total quantity of items",
            min_value=int(items_min),
            max_value=int(items_max),
            value=int(items_med),
            step=1,
        )
        n_unique_items_val = st.number_input(
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

        submitted = st.form_submit_button("Estimate coupon usage probability")

    if submitted:
        try:
            # Construct single-row DataFrame with exactly SELECTED_FEATURES
            single_row = pd.DataFrame([{
                "total_amount": float(total_amount_val),
                "n_items": int(n_items_val),
                "n_unique_items": int(n_unique_items_val),
                "cust_age": float(cust_age),
                "cust_gender": cust_gender,
                "storeLocation": store_location,
                "purchaseMethod": purchase_method,
                "sale_month": int(sale_month),
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
                "This prediction uses the same trained pipeline as in the model evaluation. "
                "Training and threshold tuning were done offline; this app only performs inference."
            )
        except Exception as e:
            st.error(f"Error computing manual prediction: {e}")
