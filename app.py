import streamlit as st
from src import data_processing, model, explainability, nlp_analysis, strategy_simulator, database, anomaly_monitor
import os
import pandas as pd
import matplotlib.pyplot as plt

# File paths (update paths as needed)
RAW_DATA_PATH = 'data/raw/player_data_enhanced.csv'
PROCESSED_DATA_PATH = 'data/processed/player_data_processed.csv'
REVIEWS_PATH = 'data/processed/reviews.csv'
MODEL_PATH = 'models/rf_model.joblib'

def main():
    st.title("🎮 Player Retention Analysis Major Project")

    menu = ["Data Processing & EDA", "Model Training & Evaluation", "Explainability (SHAP)",
            "NLP Review Analysis", "Retention Strategy Simulator", "DAU Anomaly Monitor", "SQL Data Query"]
    choice = st.sidebar.selectbox("Choose a Module", menu)

    if choice == "Data Processing & EDA":
        st.header("Data Processing & Exploratory Data Analysis (EDA)")
        if st.button("Run Data Processing Pipeline"):
            df = data_processing.load_and_process(RAW_DATA_PATH, PROCESSED_DATA_PATH)
            st.success("Data processed and saved!")
        if os.path.exists(PROCESSED_DATA_PATH):
            df = data_processing.load_raw_data(PROCESSED_DATA_PATH)
            st.write(df.head())
            data_processing.show_eda(df)
        else:
            st.warning("Processed data not found. Run pipeline above.")

    elif choice == "Model Training & Evaluation":
        st.header("Train and Evaluate Churn Prediction Model")
        if os.path.exists(PROCESSED_DATA_PATH):
            df = data_processing.load_raw_data(PROCESSED_DATA_PATH)
            if st.button("Train Model"):
                with st.spinner("Training Random Forest..."):
                    model_, X_test, y_test, y_pred, report, cm = model.train_and_evaluate(
                        df, use_grid_search=False, model_path=MODEL_PATH
                    )
                st.success("Model trained and saved.")

                st.subheader("Classification Report")
                st.dataframe(pd.DataFrame(report).transpose())

                st.subheader("Confusion Matrix")
                fig, ax = plt.subplots(figsize=(5, 4))
                from sklearn.metrics import ConfusionMatrixDisplay
                ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Churn", "Churn"]).plot(ax=ax)
                st.pyplot(fig)

                st.subheader("Feature Importance")
                importance_df = pd.DataFrame({
                    "feature": X_test.columns,
                    "importance": model_.feature_importances_
                }).sort_values("importance", ascending=False).head(10)
                st.bar_chart(importance_df.set_index("feature"))
        else:
            st.warning("Processed data not found. Please process data first.")

    elif choice == "Explainability (SHAP)":
        st.header("Model Explainability with SHAP")
        if os.path.exists(MODEL_PATH) and os.path.exists(PROCESSED_DATA_PATH):
            model_ = model.load_model(MODEL_PATH)
            df = data_processing.load_raw_data(PROCESSED_DATA_PATH)
            X = df.drop(columns=['player_id', 'churn'])
            explainability.shap_summary_plot(model_, X)
        else:
            st.warning("Model or processed data missing. Train model first.")

    elif choice == "NLP Review Analysis":
        st.header("Player Review NLP Analysis")
        if os.path.exists(REVIEWS_PATH):
            nlp_analysis.run_nlp_dashboard(REVIEWS_PATH)
        else:
            st.warning("Reviews dataset not found.")

    elif choice == "Retention Strategy Simulator":
        strategy_simulator.run_simulator()

    elif choice == "DAU Anomaly Monitor":
        st.header("DAU Anomaly Monitor")
        st.caption("Simulated 30-day DAU time series with Z-score anomaly detection.")

        anomaly_dates = st.multiselect(
            "Inject artificial anomalies (day index → DAU multiplier)",
            options=[(i, f"Day {i}: x0.4 (sharp drop)") for i in range(5, 30, 7)]
            + [(i, f"Day {i}: x0.6 (moderate drop)") for i in range(8, 28, 5)],
            default=[],
        )
        anomaly_map = {}
        for entry in anomaly_dates:
            day_idx = int(entry[0])
            anomaly_map[day_idx] = 0.4

        if st.button("Generate DAU Data & Detect Anomalies"):
            if os.path.exists(PROCESSED_DATA_PATH):
                df_players = data_processing.load_raw_data(PROCESSED_DATA_PATH)
            else:
                df_players = None

            with st.spinner("Simulating DAU and running anomaly detection..."):
                dau_df = anomaly_monitor.simulate_dau_series(df_players, days=30, anomaly_dates=anomaly_map)

            st.subheader("DAU Trend (Z-Score Method)")
            import plotly.express as px
            fig = px.line(dau_df, x="date", y="dau", title="Daily Active Users (30 days)")
            anomaly_points = dau_df[dau_df["is_anomaly"]]
            if not anomaly_points.empty:
                fig.add_scatter(
                    x=anomaly_points["date"],
                    y=anomaly_points["dau"],
                    mode="markers",
                    marker=dict(color="red", size=12, symbol="x"),
                    name="Anomaly",
                )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Z-Score Chart")
            fig2 = px.bar(
                dau_df,
                x="date",
                y="z_score",
                color="severity",
                color_discrete_map={"normal": "gray", "warning": "orange", "critical": "red"},
                title="Z-Score per Day (|Z|>2 = warning, |Z|>3 = critical)",
            )
            fig2.add_hline(y=2, line_dash="dash", line_color="orange", annotation_text="warning +2")
            fig2.add_hline(y=-2, line_dash="dash", line_color="orange")
            fig2.add_hline(y=3, line_dash="dash", line_color="red", annotation_text="critical +3")
            fig2.add_hline(y=-3, line_dash="dash", line_color="red")
            st.plotly_chart(fig2, use_container_width=True)

            st.subheader("Detected Anomalies")
            summary = anomaly_monitor.get_anomaly_summary(dau_df)
            if summary.empty:
                st.success("No anomalies detected in the current data.")
            else:
                st.dataframe(summary, use_container_width=True)

                st.subheader("Interpretation Guide")
                st.markdown("""
                | Z-Score 范围 | 严重度 | 含义 |
                |---|---|---|
                | \\|Z\\| < 2 | Normal | 正常波动范围 |
                | 2 \\u2264 \\|Z\\| < 3 | Warning | 值得关注，可能是系统性问题前兆 |
                | \\|Z\\| \\u2265 3 | Critical | 严重异常，需立即排查 |
                """)

            st.subheader("Raw Data")
            st.dataframe(dau_df, use_container_width=True)

    elif choice == "SQL Data Query":
        st.header("SQL Data Query")

        if st.button("Initialize / Refresh Database"):
            conn = database.init_db()
            st.session_state["db_conn"] = conn
            st.success("Database initialized. Player data and reviews loaded.")

        if "db_conn" not in st.session_state:
            st.info("Click the button above to initialize the database.")
        else:
            conn = st.session_state["db_conn"]
            schema = database.get_schema(conn)

            st.subheader("Tables & Columns")
            for table, cols in schema.items():
                col_str = ", ".join(f"{name} ({dtype})" for name, dtype in cols)
                st.caption(f"**{table}**  ({len(cols)} columns)")
                st.code(col_str, language=None)

            st.subheader("Write SQL Query")
            st.caption("Only SELECT queries are allowed.")

            example_queries = {
                "--- 选一个示例 ---": "",
                "流失 vs 留存概况": "SELECT churn, COUNT(*) AS cnt, ROUND(AVG(num_sessions),1) AS avg_sessions, ROUND(AVG(total_playtime),1) AS avg_playtime FROM player_data GROUP BY churn",
                "按国家看流失分布": "SELECT country_Japan, country_USA, country_UK, country_India, churn, COUNT(*) AS cnt FROM player_data GROUP BY country_Japan, country_USA, country_UK, country_India, churn ORDER BY cnt DESC",
                "按设备看付费": "SELECT device_Mobile, device_PC, ROUND(AVG(in_game_purchases),2) AS avg_purchases FROM player_data GROUP BY device_Mobile, device_PC",
                "高活跃未流失玩家": "SELECT player_id, num_sessions, total_playtime, days_since_last_session FROM player_data WHERE churn=0 AND num_sessions > 15 ORDER BY num_sessions DESC LIMIT 10",
                "近期不活跃高风险玩家": "SELECT player_id, days_since_last_session, num_sessions, churn FROM player_data WHERE days_since_last_session > 20 ORDER BY days_since_last_session DESC LIMIT 10",
                "评论情感分布": "SELECT * FROM player_reviews LIMIT 10",
            }

            selected_example = st.selectbox("Example queries", list(example_queries.keys()))
            if selected_example != "--- 选一个示例 ---":
                sql_input = st.text_area("SQL", value=example_queries[selected_example], height=100)
            else:
                sql_input = st.text_area("SQL", height=100, placeholder="SELECT * FROM player_data LIMIT 5")

            if st.button("Run Query"):
                if not sql_input.strip():
                    st.warning("Please enter a SQL query.")
                else:
                    success, result, columns = database.run_query(conn, sql_input)
                    if success:
                        st.success(f"Returned {len(result)} rows")
                        st.dataframe(result)
                    else:
                        st.error(result)

if __name__ == "__main__":
    main()
