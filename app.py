import streamlit as st
from src import data_processing, model, explainability, nlp_analysis, strategy_simulator, database
import os

# File paths (update paths as needed)
RAW_DATA_PATH = 'data/raw/player_data_enhanced.csv'
PROCESSED_DATA_PATH = 'data/processed/player_data_processed.csv'
REVIEWS_PATH = 'data/processed/reviews.csv'
MODEL_PATH = 'models/rf_model.joblib'

def main():
    st.title("🎮 Player Retention Analysis Major Project")

    menu = ["Data Processing & EDA", "Model Training & Evaluation", "Explainability (SHAP)",
            "NLP Review Analysis", "Retention Strategy Simulator", "SQL Data Query"]
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
                model_, X_test = model.train_and_evaluate(df, use_grid_search=False, model_path=MODEL_PATH)
                st.success("Model trained and saved.")
                st.write("Test data sample:")
                st.write(X_test.head())
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
