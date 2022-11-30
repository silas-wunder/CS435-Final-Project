import streamlit as st
import pandas as pd
import numpy as np

steam_df = pd.read_parquet("time-series/finished_data.parquet")

st.write("# Steam Dataset Anaylsis")
st.write("## Number of Languages related to Monthly ")
value = st.slider("Test Slider", 0, 10)
test_df = pd.DataFrame(data={"Amazing": np.arange(1, value)})
test_df