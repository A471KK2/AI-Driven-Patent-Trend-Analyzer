import os
import streamlit as st

from information_collector import fetch_patent_data
from ingestion import ingest_from_directory
from patent_crew import run_patent_analysis


st.set_page_config(page_title="Patent Innovation Predictor", layout="wide")

st.title("🚀 AI-Driven-Trend-Analyzer")

st.markdown("Analyze patents and predict future innovations using AI")

# ----------------------------
# USER INPUTS
# ----------------------------
query = st.text_input("Enter research topic", value="cars")
model_name = st.text_input("Ollama model", value="deepseek-r1:1.5b")

results_dir = "results"
output_file = "latest_analysis.txt"

# ----------------------------
# BUTTON 1: LOAD PATENTS
# ----------------------------
st.header("1️⃣ Load Patents")

if st.button("Load Patents"):
    with st.spinner("Fetching patent data..."):
        try:
            fetch_patent_data(query, results_dir)
            count = ingest_from_directory(results_dir)
            st.success(f"Loaded and indexed {count} patents.")
        except Exception as e:
            st.error(str(e))

# ----------------------------
# BUTTON 2: RUN ANALYSIS
# ----------------------------
st.header("2️⃣ Run Analysis")

if st.button("Run Analysis"):
    with st.spinner("Running patent analysis..."):
        try:
            result = run_patent_analysis(query, model_name)

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result)

            st.success("Analysis completed.")
        except Exception as e:
            st.error(str(e))

# ----------------------------
# BUTTON 3: VIEW RESULTS
# ----------------------------
st.header("3️⃣ View Results")

if os.path.exists(output_file):
    with open(output_file, "r", encoding="utf-8") as f:
        content = f.read()

    st.text_area("Analysis Output", content, height=400)

    st.download_button(
        label="Download Results",
        data=content,
        file_name="patent_analysis.txt",
        mime="text/plain"
    )
else:
    st.info("Run analysis to view results.")





