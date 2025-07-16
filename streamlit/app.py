import streamlit as st
import pandas as pd
import joblib
from feature_engineering import engineer_features
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Clickstream Customer Analysis", layout="wide")

st.title(" Clickstream Customer Conversion Prediction App")

uploaded_file = st.file_uploader(" Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader(" Raw Uploaded Data")
    st.dataframe(df.head())

    df_fe = engineer_features(df)

    drop_cols = ['year', 'session_id', 'converted', 'revenue'] if 'converted' in df_fe.columns and 'revenue' in df_fe.columns else ['year', 'session_id']
    X = df_fe.drop(columns=[col for col in drop_cols if col in df_fe.columns], errors='ignore')

    clf_model = joblib.load("notebooks/best_classification_pipeline.pkl")

    y_pred_class = clf_model.predict(X)
    y_prob_class = clf_model.predict_proba(X)[:, 1]

    st.subheader(" Conversion Prediction")
    df['predicted_conversion'] = y_pred_class
    df['conversion_probability'] = y_prob_class
    st.dataframe(df[['session_id', 'predicted_conversion', 'conversion_probability']].head())

    df_converted = df_fe[y_pred_class == 1]
    if not df_converted.empty:
        X_reg = df_converted.drop(columns=drop_cols, errors='ignore')
        reg_model = joblib.load("notebooks/best_regression_model.pkl")
        y_pred_rev = reg_model.predict(X_reg)

        st.subheader(" Revenue Prediction (For Converted Users)")
        df_converted_result = df[df['predicted_conversion'] == 1].copy()
        df_converted_result['predicted_revenue'] = y_pred_rev
        st.dataframe(df_converted_result[['session_id', 'predicted_revenue']].head())
    else:
        st.warning("No predicted converted users to estimate revenue.")

    st.subheader(" Customer Segmentation (Clustering)")
    cluster_model = joblib.load("notebooks/kmeans_clustering_model.pkl")

    cluster_cols = ['price', 'session_length', 'num_clicks', 'category_1_clicks', 'category_2_clicks',
                    'category_3_clicks', 'category_4_clicks', 'exit_page', 'is_bounce', 'repeated_views']

    missing_cols = set(cluster_cols) - set(df_fe.columns)
    if missing_cols:
        st.error(f" Missing columns for clustering: {missing_cols}")
    else:
        X_cluster = df_fe[cluster_cols].copy()

        X_cluster['exit_page'] = X_cluster['exit_page'].astype('category').cat.codes

        scaler = StandardScaler()
        X_cluster_scaled = scaler.fit_transform(X_cluster)

        df['cluster'] = cluster_model.predict(X_cluster_scaled)
        st.dataframe(df[['session_id', 'cluster']].head())

        st.bar_chart(df['cluster'].value_counts().sort_index())

else:
    st.info("Please upload a CSV file to proceed.")
