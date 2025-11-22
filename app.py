import os
import json

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

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

# Features used by the deployed model (internal names)
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
        raise RuntimeError("Environment variables COSMOS_URL or COSMOS_KEY are not set.")
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
    used by the deployed model.
    """
    df = df_raw.copy()

    if TARGET_COL not in df.columns:
        raise KeyError(f"Column '{TARGET_COL}' must be present in the raw data.")
    if "saleDate" not in df.columns:
        raise KeyError("Column 'saleDate' must be present in the raw data.")

    # Keep only rows with known target
    df = df[df[TARGET_COL].notna()].copy()
    df[TARGET_COL] = df[TARGET_COL].astype(bool)

    # Parse date
    df["saleDate"] = pd.to_datetime(df["saleDate"], errors="coerce")

    # Basket-level features
    df["total_amount"] = df["items"].apply(total_amount)
    df["n_items"] = df["items"].apply(total_quantity)
    df["n_unique_items"] = df["items"].apply(n_unique_items)

    # Customer features
    df["cust_age"] = df["customer"].apply(lambda c: get_customer_field(c, "age"))
    df["cust_gender"] = df["customer"].apply(lambda c: get_customer_field(c, "gender"))

    # Time features
    df["sale_month"] = df["saleDate"].dt.month

    cols_needed = SELECTED_FEATURES + [TARGET_COL]
    df_model = df[cols_needed].copy()

    # Clean infinities and drop rows with any missing feature or target
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
    Load the deployed sklearn pipeline and the tuned decision threshold.
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
    Load model comparison metrics from JSON.
    """
    if not os.path.exists(RESULTS_PATH):
        return None
    with open(RESULTS_PATH, "r") as f:
        return json.load(f)


def compute_metrics(y_true, y_prob, threshold):
    """
    Compute common classification metrics at a given decision threshold.
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
    """
    df_sales = load_sales_data()
    df_model = build_feature_table(df_sales)

    X_all = df_model[SELECTED_FEATURES].copy()
    y_all = df_model[TARGET_COL].values

    n_total = len(df_model)
    n_valid_feat = len(df_model)
    return X_all, y_all, n_total, n_valid_feat, df_model


# -------------------------------------------------------------------
# Streamlit layout & small typography tweaks
# -------------------------------------------------------------------

st.set_page_config(
    page_title="Coupon Usage Prediction – Cosmos DB & ML",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Just bump heading & metric sizes a bit, no color overrides
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1450px;
        margin: 0 auto;
    }
    h1 {
        font-size: 2.4rem !important;
        font-weight: 700 !important;
    }
    h2 {
        font-size: 1.9rem !important;
        font-weight: 650 !important;
    }
    h3 {
        font-size: 1.6rem !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 1.0rem !important;
        font-weight: 500 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Coupon Usage Prediction – Retail Insights Dashboard")

st.markdown(
    """
    This dashboard connects **Azure Cosmos DB** with **machine learning models** to show:
    - How often coupons are used and where  
    - How coupon usage varies over time and across stores  
    - A predictive model that estimates the probability of coupon usage for a new sale  
    """
)

tab_overview, tab_data, tab_imbalance, tab_models, tab_inference = st.tabs(
    [
        "Executive Overview",
        "Detailed Exploration",
        "Target Balance",
        "Model Comparison",
        "Deployed Model & What-If",
    ]
)

# -------------------------------------------------------------------
# Tab 1 – Executive Overview
# -------------------------------------------------------------------

with tab_overview:
    st.header("1. Executive Overview")

    try:
        df_sales = load_sales_data()
        df_model = build_feature_table(df_sales)
    except Exception as e:
        st.error(f"Error loading or processing data from Cosmos DB: {e}")
        st.stop()

    # KPIs
    n_tx = len(df_model)
    overall_coupon_rate = df_model[TARGET_COL].mean() if n_tx > 0 else 0.0
    n_stores = df_model["storeLocation"].nunique() if "storeLocation" in df_model.columns else 0
    avg_basket = df_model["total_amount"].mean() if "total_amount" in df_model.columns else 0.0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Model-ready Transactions", f"{n_tx:,}")
    with k2:
        st.metric("Overall Coupon Usage Rate", f"{overall_coupon_rate*100:,.1f}%")
    with k3:
        st.metric("Stores in Dataset", f"{n_stores}")
    with k4:
        st.metric("Average Basket Value", f"${avg_basket:,.2f}")

    st.markdown("---")

    # Coupon rate by month (Altair line chart)
    st.subheader("Coupon Usage Over Time (by Month)")
    if {"sale_month", TARGET_COL}.issubset(df_model.columns):
        month_agg = (
            df_model.groupby("sale_month")[TARGET_COL]
            .agg(Coupon_Usage_Rate="mean", Number_of_Transactions="count")
            .reset_index()
        )
        month_agg["Month"] = month_agg["sale_month"].astype(int)

        line_chart = (
            alt.Chart(month_agg)
            .mark_line(point=True)
            .encode(
                x=alt.X("Month:O", title="Month"),
                y=alt.Y("Coupon_Usage_Rate:Q", title="Coupon usage rate"),
            )
            .properties(height=300)
        )
        st.altair_chart(line_chart, use_container_width=True)
        st.caption(
            "Coupon usage rate by month (1 = January, 12 = December). "
            "Peaks suggest periods where customers are more responsive to promotions."
        )
    else:
        st.info("Cannot compute monthly view – required time fields are missing.")

    # Coupon rate by store (Altair bar chart)
    st.subheader("Coupon Usage by Store (Top 10 Locations)")

    if {"storeLocation", TARGET_COL}.issubset(df_model.columns):
        store_agg = (
            df_model.groupby("storeLocation")[TARGET_COL]
            .agg(Coupon_Usage_Rate="mean", Number_of_Transactions="count")
            .reset_index()
        )
        store_agg = store_agg.sort_values("Number_of_Transactions", ascending=False).head(10)

        bar_chart = (
            alt.Chart(store_agg)
            .mark_bar()
            .encode(
                x=alt.X("Coupon_Usage_Rate:Q", title="Coupon usage rate"),
                y=alt.Y("storeLocation:N", sort="-x", title="Store"),
            )
            .properties(height=350)
        )
        st.altair_chart(bar_chart, use_container_width=True)
        st.caption(
            "Top stores by transaction volume and their coupon usage rate. "
            "Locations with higher rates may be more promotion-sensitive."
        )
    else:
        st.info("Cannot compute store view – missing location or target fields.")

    st.subheader("Key Takeaways")
    st.markdown(
        """
        - **Overall usage:** The coupon usage rate gives a baseline for how often customers
          take advantage of promotions.
        - **Seasonality:** The month-by-month line chart highlights potential **campaign windows**.
        - **Store differences:** The store bar chart shows where coupon behavior is stronger or weaker,
          guiding local or regional strategies.
        """
    )

# -------------------------------------------------------------------
# Tab 2 – Detailed Exploration (EDA with filters)
# -------------------------------------------------------------------

with tab_data:
    st.header("2. Detailed Data Exploration")

    try:
        df_sales = load_sales_data()
        df_model = build_feature_table(df_sales)
        st.success(f"Model-ready transactions after cleaning: **{len(df_model)}**")
    except Exception as e:
        st.error(f"Error loading or processing data: {e}")
        df_model = None

    if df_model is None or df_model.empty:
        st.warning("Feature table is empty; cannot perform exploration.")
    else:
        # Filters
        st.subheader("Interactive Filters")

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
            f1, f2, f3 = st.columns(3)

            with f1:
                store_filter = st.multiselect(
                    "Store location(s)",
                    options=stores_all,
                    default=stores_all,
                )

            with f2:
                method_filter = st.multiselect(
                    "Sales channel(s)",
                    options=methods_all,
                    default=methods_all,
                )

            with f3:
                month_range = st.slider(
                    "Sale month range",
                    min_value=int(default_month_min),
                    max_value=int(default_month_max),
                    value=(int(default_month_min), int(default_month_max)),
                    step=1,
                )

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
            f"Filtered subset: **{len(df_filt)}** transactions "
            f"(starting from {len(df_model)} model-ready records)."
        )

        st.markdown("#### Sample of Filtered Data")
        st.dataframe(
            df_filt.head(20).rename(
                columns={
                    "total_amount": "BasketValue",
                    "n_items": "TotalItems",
                    "n_unique_items": "DistinctProducts",
                    "cust_age": "CustomerAge",
                    "cust_gender": "CustomerGender",
                    "storeLocation": "Store",
                    "purchaseMethod": "Channel",
                    "sale_month": "SaleMonth",
                    TARGET_COL: "CouponUsedFlag",
                }
            ),
            use_container_width=True,
        )

        if df_filt.empty:
            st.warning("No rows match the selected filters. Adjust filters to explore the data.")
        else:
            st.subheader("Basket & Customer Profile (Filtered Subset)")

            num_cols_for_desc = ["total_amount", "n_items", "n_unique_items", "cust_age"]
            num_cols_for_desc = [c for c in num_cols_for_desc if c in df_filt.columns]

            if num_cols_for_desc:
                desc = df_filt[num_cols_for_desc].describe().T.reset_index()
                desc = desc.rename(
                    columns={
                        "index": "Feature",
                        "mean": "Mean",
                        "std": "Std Dev",
                        "min": "Min",
                        "max": "Max",
                    }
                )
                cols_to_show = [c for c in ["Feature", "Mean", "Std Dev", "Min", "Max"] if c in desc.columns]
                st.dataframe(desc[cols_to_show])
            else:
                st.info("No numeric columns available for summary statistics.")

            c1, c2 = st.columns(2)

            with c1:
                st.markdown("#### Transactions by Store (Filtered)")
                if "storeLocation" in df_filt.columns:
                    store_counts = (
                        df_filt["storeLocation"]
                        .value_counts()
                        .rename_axis("Store")
                        .to_frame("Number_of_Transactions")
                        .reset_index()
                    )
                    chart_store = (
                        alt.Chart(store_counts)
                        .mark_bar()
                        .encode(
                            x=alt.X("Number_of_Transactions:Q", title="Transactions"),
                            y=alt.Y("Store:N", sort="-x"),
                        )
                        .properties(height=300)
                    )
                    st.altair_chart(chart_store, use_container_width=True)
                else:
                    st.info("Store information is not available in this dataset.")

            with c2:
                st.markdown("#### Transactions by Channel (Filtered)")
                if "purchaseMethod" in df_filt.columns:
                    channel_counts = (
                        df_filt["purchaseMethod"]
                        .value_counts()
                        .rename_axis("Channel")
                        .to_frame("Number_of_Transactions")
                        .reset_index()
                    )
                    chart_channel = (
                        alt.Chart(channel_counts)
                        .mark_bar()
                        .encode(
                            x=alt.X("Number_of_Transactions:Q", title="Transactions"),
                            y=alt.Y("Channel:N", sort="-x"),
                        )
                        .properties(height=300)
                    )
                    st.altair_chart(chart_channel, use_container_width=True)
                else:
                    st.info("Channel information is not available in this dataset.")

            st.markdown("#### Basket Size & Coupon Usage (Filtered)")

            if TARGET_COL in df_filt.columns:
                agg = (
                    df_filt.groupby(TARGET_COL)
                    .agg(
                        avg_total_amount=("total_amount", "mean"),
                        avg_n_items=("n_items", "mean"),
                        avg_n_unique_items=("n_unique_items", "mean"),
                    )
                    .reset_index()
                )
                agg["CouponUsage"] = agg[TARGET_COL].map({0: "No coupon used", 1: "Coupon used"})
                agg_pretty = agg.rename(
                    columns={
                        "avg_total_amount": "Average Basket Value",
                        "avg_n_items": "Average Number of Items",
                        "avg_n_unique_items": "Average Number of Distinct Products",
                    }
                )[["CouponUsage", "Average Basket Value", "Average Number of Items", "Average Number of Distinct Products"]]
                st.dataframe(agg_pretty.set_index("CouponUsage"))
                st.caption(
                    "Comparison of basket characteristics for transactions with and without coupon usage."
                )
            else:
                st.info("Cannot compute basket comparison – coupon flag not available.")

            st.markdown("#### Coupon Usage Over Time (Filtered Subset)")
            if {"sale_month", TARGET_COL}.issubset(df_filt.columns):
                month_agg_f = (
                    df_filt.groupby("sale_month")[TARGET_COL]
                    .agg(Coupon_Usage_Rate="mean", Number_of_Transactions="count")
                    .reset_index()
                )
                month_agg_f["Month"] = month_agg_f["sale_month"].astype(int)
                chart_month_f = (
                    alt.Chart(month_agg_f)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("Month:O", title="Month"),
                        y=alt.Y("Coupon_Usage_Rate:Q", title="Coupon usage rate"),
                    )
                    .properties(height=280)
                )
                st.altair_chart(chart_month_f, use_container_width=True)
            else:
                st.info("Cannot compute monthly trend – missing date or target information.")

            st.markdown("#### Coupon Usage by Month & Store (Filtered Subset)")
            if {"sale_month", "storeLocation", TARGET_COL}.issubset(df_filt.columns):
                store_month_agg = (
                    df_filt.groupby(["sale_month", "storeLocation"])[TARGET_COL]
                    .mean()
                    .reset_index()
                )
                store_month_agg["Month"] = store_month_agg["sale_month"].astype(int)
                chart_store_month = (
                    alt.Chart(store_month_agg)
                    .mark_line()
                    .encode(
                        x=alt.X("Month:O", title="Month"),
                        y=alt.Y(f"{TARGET_COL}:Q", title="Coupon usage rate"),
                        color=alt.Color("storeLocation:N", title="Store"),
                    )
                    .properties(height=320)
                )
                st.altair_chart(chart_store_month, use_container_width=True)
                st.caption(
                    "Each line represents a store’s coupon usage rate over the months, "
                    "within the selected filter context."
                )
            else:
                st.info(
                    "Cannot compute store-by-month view – missing month, store, or coupon fields."
                )

# -------------------------------------------------------------------
# Tab 3 – Target Balance
# -------------------------------------------------------------------

with tab_imbalance:
    st.header("3. Target Balance – How Often Are Coupons Used?")

    df_sales = load_sales_data()

    if TARGET_COL in df_sales.columns:
        vc = df_sales[TARGET_COL].value_counts(dropna=False)
        vc_named = vc.rename(index={0: "No coupon used (0)", 1: "Coupon used (1)"})
        df_vc = vc_named.reset_index().rename(
            columns={"index": "CouponUsageFlag", TARGET_COL: "Number_of_Transactions"}
        )

        st.markdown("#### Distribution of Coupon Usage Flag (Full Raw Data)")
        st.dataframe(df_vc, use_container_width=True)

        chart_vc = (
            alt.Chart(df_vc)
            .mark_bar()
            .encode(
                x=alt.X("CouponUsageFlag:N", title="Coupon usage flag"),
                y=alt.Y("Number_of_Transactions:Q", title="Number of transactions"),
            )
            .properties(height=300)
        )
        st.altair_chart(chart_vc, use_container_width=True)

        st.markdown(
            """
            The dataset is **imbalanced**:
            - Most transactions fall under “no coupon used”.
            - A smaller proportion corresponds to “coupon used”.

            During modeling, this imbalance was addressed by:
            - Adjusting **class weights** in the models.
            - Tuning the **decision threshold** to get a better F1-score for the “coupon used” class.
            """
        )
    else:
        st.warning(
            f"Column '{TARGET_COL}' is not present in the raw data; cannot display target balance."
        )

# -------------------------------------------------------------------
# Tab 4 – Model Comparison
# -------------------------------------------------------------------

with tab_models:
    st.header("4. Model Comparison – Validation & Test")

    results_json = load_model_results()
    if results_json is None:
        st.error(
            "`model_results.json` not found. Please include the `models/` folder generated by the training notebook."
        )
    else:
        baseline = pd.DataFrame(results_json.get("baseline", []))
        improved = results_json.get("improved_best_sklearn", None)

        st.subheader("4.1 Validation Results (Each Model at Its Best Threshold)")
        st.markdown(
            """
            Each candidate model is evaluated on the validation set at the threshold that maximizes
            its F1-score for the coupon usage class.
            """
        )
        st.dataframe(baseline, use_container_width=True)

        st.subheader("4.2 Deployed Model – Summary")

        if improved is not None:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Validation (Best Threshold for This Model)**")
                st.json(improved["baseline"])
            with c2:
                st.markdown("**Test Set (Same Threshold on Unseen Data)**")
                st.json(improved["improved"])
        else:
            st.info(
                "Improved metrics for the deployed model were not found in `model_results.json`."
            )

        st.subheader("4.3 Confusion Matrices on Current Cosmos Data")

        try:
            pipeline, best_threshold = load_model_and_threshold()
            X_all, y_all, n_total, n_valid_feat, df_model_all = prepare_labeled_data_for_pipeline()

            if X_all is None or len(X_all) == 0:
                st.info("No labeled, model-ready rows to compute confusion matrices.")
            else:
                y_prob_all = pipeline.predict_proba(X_all)[:, 1]

                acc_05, auc_05, prec_05, rec_05, f1_05, cm_05 = compute_metrics(
                    y_all, y_prob_all, 0.5
                )
                acc_t, auc_t, prec_t, rec_t, f1_t, cm_t = compute_metrics(
                    y_all, y_prob_all, best_threshold
                )

                d1, d2 = st.columns(2)
                with d1:
                    st.markdown("**Using the default threshold = 0.5**")
                    df_cm_05 = pd.DataFrame(
                        cm_05,
                        index=["True: No coupon", "True: Coupon used"],
                        columns=["Predicted: No coupon", "Predicted: Coupon used"],
                    )
                    st.table(df_cm_05)
                    st.caption(
                        f"Accuracy={acc_05:.3f}, Precision={prec_05:.3f}, "
                        f"Recall={rec_05:.3f}, F1={f1_05:.3f}"
                    )

                with d2:
                    st.markdown(f"**Using the tuned threshold = {best_threshold:.2f}**")
                    df_cm_t = pd.DataFrame(
                        cm_t,
                        index=["True: No coupon", "True: Coupon used"],
                        columns=["Predicted: No coupon", "Predicted: Coupon used"],
                    )
                    st.table(df_cm_t)
                    st.caption(
                        f"Accuracy={acc_t:.3f}, Precision={prec_t:.3f}, "
                        f"Recall={rec_t:.3f}, F1={f1_t:.3f}"
                    )

        except Exception as e:
            st.error(f"Error computing confusion matrices for the deployed model: {e}")

# -------------------------------------------------------------------
# Tab 5 – Deployed Model & What-If Analysis
# -------------------------------------------------------------------

with tab_inference:
    st.header("5. Deployed Model & What-If Analysis")

    st.markdown(
        """
        The deployed model uses:
        - Basket value and size (total amount, number of items, number of distinct products)
        - Customer age and gender
        - Store location
        - Sales channel (in-store, online, phone)
        - Month of the sale

        to estimate the **probability that a coupon is used** for a transaction.
        """
    )

    try:
        pipeline, best_threshold = load_model_and_threshold()
        st.success(
            f"Deployed pipeline loaded from `{MODEL_PATH}` "
            f"with tuned threshold **{best_threshold:.2f}**."
        )
    except Exception as e:
        st.error(f"Error loading the deployed model: {e}")
        st.stop()

    # --- Part A: Batch evaluation on current data ---

    st.subheader("5.1 Performance on Current Data (Global View)")

    try:
        X_all, y_all, n_total, n_valid_feat, df_model_all = prepare_labeled_data_for_pipeline()
    except Exception as e:
        st.error(f"Error preparing data for evaluation: {e}")
        X_all, y_all = None, None

    if X_all is None or len(X_all) == 0:
        st.info("There are no model-ready rows to evaluate the deployed model.")
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

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Accuracy", f"{acc:.3f}")
            st.metric("ROC-AUC", f"{auc:.3f}" if not np.isnan(auc) else "N/A")
        with m2:
            st.metric("Precision (Coupon = 1)", f"{prec:.3f}")
            st.metric("Recall (Coupon = 1)", f"{rec:.3f}")
        with m3:
            st.metric("F1-Score (Coupon = 1)", f"{f1:.3f}")

        st.markdown("#### Confusion Matrix at Selected Threshold")
        df_cm = pd.DataFrame(
            cm,
            index=["True: No coupon", "True: Coupon used"],
            columns=["Predicted: No coupon", "Predicted: Coupon used"],
        )
        st.table(df_cm)

    # --- Part B: What-If form (single transaction) ---

    st.subheader("5.2 What-If Scenario – Single Transaction Simulation")

    # Build df_model to derive realistic ranges and categories
    try:
        df_sales = load_sales_data()
        df_model = build_feature_table(df_sales)
    except Exception as e:
        st.error(f"Error rebuilding feature table for what-if defaults: {e}")
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

    # Fallback options if dataset is empty
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
            "Basket value (total amount)",
            min_value=float(amt_min),
            max_value=float(amt_max),
            value=float(amt_med),
            step=10.0,
        )
        n_items_val = st.number_input(
            "Number of items in the basket",
            min_value=int(items_min),
            max_value=int(items_max),
            value=int(items_med),
            step=1,
        )
        n_unique_items_val = st.number_input(
            "Number of different products",
            min_value=int(uniq_min),
            max_value=int(uniq_max),
            value=int(uniq_med),
            step=1,
        )

        st.markdown("**Store & channel**")
        store_location = st.selectbox("Store location", store_locations)
        purchase_method = st.selectbox("Sales channel", purchase_methods)

        st.markdown("**Timing**")
        sale_month = st.slider(
            "Month of the sale (1 = January, 12 = December)",
            min_value=1,
            max_value=12,
            value=int(default_month),
        )

        submitted = st.form_submit_button("Estimate coupon usage probability")

    if submitted:
        try:
            single_row = pd.DataFrame(
                [
                    {
                        "total_amount": float(total_amount_val),
                        "n_items": int(n_items_val),
                        "n_unique_items": int(n_unique_items_val),
                        "cust_age": float(cust_age),
                        "cust_gender": cust_gender,
                        "storeLocation": store_location,
                        "purchaseMethod": purchase_method,
                        "sale_month": int(sale_month),
                    }
                ]
            )

            prob_coupon = pipeline.predict_proba(single_row)[:, 1][0]
            pred_label = int(prob_coupon >= best_threshold)

            st.markdown("#### Prediction result")
            st.write(f"Estimated probability of coupon usage: **{prob_coupon:.3f}**")
            st.write(f"Decision threshold in use: **{best_threshold:.2f}**")

            if pred_label == 1:
                st.success("Model decision: **Coupon usage likely (class 1).**")
            else:
                st.info("Model decision: **Coupon usage unlikely (class 0).**")

            st.caption(
                "This scenario uses the deployed model exactly as in production: "
                "the model and threshold are pre-trained; the dashboard only performs inference."
            )

        except Exception as e:
            st.error(f"Error computing prediction for the what-if scenario: {e}")
