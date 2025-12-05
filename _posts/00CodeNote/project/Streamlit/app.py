import streamlit as st
import pandas as pd

st.title("🍎 Fruit Price Tracker")

# Load CSV
df = pd.read_csv("./fruit_prices.csv")

st.subheader("Raw Price Table")
st.dataframe(df, use_container_width=True)

# Optional: Pivot table for better comparison
pivot_df = df.pivot(index="fruit", columns="shop", values="price")

st.subheader("Price Comparison Across Shops")
st.dataframe(pivot_df, use_container_width=True)
