import streamlit as st
import pandas as pd
import altair as alt

# ----------- Custom Theme -----------
st.set_page_config(
    page_title="Fruit Price Tracker",
    page_icon="🍓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------- Light/Dark Mode Toggle -----------
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=False)

if dark_mode:
    # Dark theme (CSS injection)
    st.markdown(
        """
        <style>
        body {
            background-color: #1e1e1e;
            color: white;
        }
        .stDataFrame { background-color: #2c2c2c; }
        .stTabs [role="tab"] { color: white !important; }
        </style>
    """,
        unsafe_allow_html=True,
    )
else:
    # Light theme
    st.markdown(
        """
        <style>
        body {
            background-color: #ffffff;
            color: black;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

# ----------- Title -----------
st.title("🍎 Fruit Price Comparison Dashboard")

# Load data
df = pd.read_csv("fruit_prices.csv")

# ----------- Sidebar filters -----------
st.sidebar.header("Filters")

fruit_list = sorted(df["fruit"].unique())
shop_list = sorted(df["shop"].unique())

selected_fruit = st.sidebar.multiselect(
    "Select Fruit", options=fruit_list, default=fruit_list
)

selected_shop = st.sidebar.multiselect(
    "Select Shop", options=shop_list, default=shop_list
)

# Filter data
filtered_df = df[(df["fruit"].isin(selected_fruit)) & (df["shop"].isin(selected_shop))]

# ----------- Tabs -----------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📄 Data Table", "📈 Charts", "📊 Shop Comparison", "🔍 Analytics"]
)

# -------------------------------------------------------------------
# Tab 1 – Data Table
# -------------------------------------------------------------------
with tab1:
    st.subheader("📄 Filtered Price Table")
    st.dataframe(filtered_df, use_container_width=True)

    pivot_df = filtered_df.pivot(index="fruit", columns="shop", values="price")
    st.subheader("📊 Pivot Comparison Table")
    st.dataframe(pivot_df, use_container_width=True)

# -------------------------------------------------------------------
# Tab 2 – Line Chart by Fruit
# -------------------------------------------------------------------
with tab2:
    st.subheader("📈 Price Trend (Line Chart)")

    line_chart = (
        alt.Chart(filtered_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("shop:N", title="Shop"),
            y=alt.Y("price:Q", title="Price ($)"),
            color="fruit:N",
        )
        .properties(height=350)
    )

    st.altair_chart(line_chart, use_container_width=True)

# -------------------------------------------------------------------
# Tab 3 – Bar Chart by Shop
# -------------------------------------------------------------------
with tab3:
    st.subheader("🏪 Shop Price Comparison (Bar Chart)")

    bar_chart = (
        alt.Chart(filtered_df)
        .mark_bar()
        .encode(
            x=alt.X("fruit:N", title="Fruit"),
            y=alt.Y("price:Q", title="Price ($)"),
            color="shop:N",
            column="shop:N",
        )
        .properties(height=350)
    )

    st.altair_chart(bar_chart, use_container_width=True)

# -------------------------------------------------------------------
# Tab 4 – Analytics
# -------------------------------------------------------------------
with tab4:
    st.subheader("🔍 Automated Pricing Analysis")

    # Average price per fruit
    avg_price = filtered_df.groupby("fruit")["price"].mean().reset_index()
    avg_price.columns = ["fruit", "average_price"]

    st.write("### 🍉 Average Price per Fruit")
    st.dataframe(avg_price, use_container_width=True)

    # Cheapest + Most Expensive shops
    st.write("### 🏆 Cheapest & Most Expensive Shops")

    cheapest = filtered_df.loc[filtered_df["price"].idxmin()]
    most_expensive = filtered_df.loc[filtered_df["price"].idxmax()]

    col1, col2 = st.columns(2)

    with col1:
        st.success(
            f"""
        🥇 **Cheapest Price**  
        - Fruit: **{cheapest['fruit']}**  
        - Shop: **{cheapest['shop']}**  
        - Price: **${cheapest['price']:.2f}**
        """
        )

    with col2:
        st.error(
            f"""
        💰 **Most Expensive Price**  
        - Fruit: **{most_expensive['fruit']}**  
        - Shop: **{most_expensive['shop']}**  
        - Price: **${most_expensive['price']:.2f}**
        """
        )

    # Summary message
    st.info(
        "📌 Summary: These insights help you quickly identify the best shop for each fruit and overall price differences."
    )
