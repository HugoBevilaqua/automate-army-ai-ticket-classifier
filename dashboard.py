import streamlit as st
import pandas as pd
import plotly.express as px
from config import MASTER_CATEGORIES, MASTER_PRIORITIES, MASTER_COLUMNS, MASTER_COLUMNS_FALLBACK, OUTPUT_FILE, CONFIDENCE_THRESHOLD, STATS_FILE

# Configure browser tab and page width
st.set_page_config(page_title="AI Ticket Insights", page_icon="📊", layout="wide")

# --- DATA LOADING ---
def load_data() -> tuple[pd.DataFrame, dict]:
    """Load enriched tickets and statistics with graceful fallback if stats unavailable."""
    # Ingests pre-processed data and statistics.
    # Uses a decoupled loading strategy: if the stats file fails, 
    # the dashboard still attempts to render the raw data. 
    try:
        # Load the primary enriched dataset
        df = pd.read_csv(OUTPUT_FILE)
        # Define an empty dict so the dashboard still works even if we fail to process any stats
        stats_dict = {}
        # Load pre-calculated metrics from analyst.py
        try:
            stats_df = pd.read_csv(STATS_FILE)
            stats_dict = dict(zip(stats_df['metric'], stats_df['value']))
        except FileNotFoundError:
            st.warning(f"Error: '{STATS_FILE}' not found. Some metrics may be unavailable.")
        
        return df, stats_dict

    except FileNotFoundError:
        st.error(f"Error: '{OUTPUT_FILE}' not found. Please run classifier.py first!")
        return pd.DataFrame(), {}

# Execution: Load data into memory
df, stats = load_data()

# Only render UI if data exists
if not df.empty:
    # --- SIDEBAR FILTERS ---
    # Allows user to closely examine specific data segments
    st.sidebar.header("Filter Options")
    
    selected_cats = st.sidebar.multiselect(
        "Select Categories",
        options=MASTER_CATEGORIES + ["Error"],
        default=MASTER_CATEGORIES + ["Error"]
    )
    
    selected_prio = st.sidebar.multiselect(
        "Select Priorities",
        options=MASTER_PRIORITIES + ["Error"],
        default=MASTER_PRIORITIES + ["Error"]
    )

    # Filtering logic: updates the dashboard
    filtered_df = df[
        (df[MASTER_COLUMNS[0]].isin(selected_cats)) & 
        (df[MASTER_COLUMNS[1]].isin(selected_prio))
    ]

    # --- MAIN DASHBOARD UI ---
    st.title("🎫 AI Support Ticket Dashboard")
    st.markdown("Analysis of customer issues using Google Gemini.")
    st.divider()

    # --- EXPORT ACTIONS ---
    # Provides buttons to download copies of enriched_tickets.csv and statistics.csv
    col1, col2, colspacer = st.columns([2,2,10])
    with col1:
        # Export Enriched CSV
        try:
            with open(OUTPUT_FILE, "rb") as file:
                st.download_button(
                    label="📥 Download Copy of Enriched Tickets",
                    data=file,
                    file_name=OUTPUT_FILE,
                    mime='text/csv',
                    help="Download the complete list of tickets with AI-enriched labels"
                )
        except FileNotFoundError:
            st.button("Tickets Missing", disabled=True)
    with col2:
        # Export statistics
        try:
            with open(STATS_FILE, "rb") as file:
                st.download_button(
                    label="📥 Download Copy of Statistics",
                    data=file,
                    file_name=STATS_FILE,
                    mime="text/csv",
                    help="Download statistics derived from the enriched tickets."
                )
        except FileNotFoundError:
            st.button("Stats Missing", disabled=True)

    st.divider()

    # --- TOP-LEVEL METRICS ---
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Processed", int(stats.get('total_processed', len(df))))
    with col2:
        st.metric("Total Failed", int(stats.get('error_count', 0)))        
    with col3:
        st.metric("Matching Filter", len(filtered_df))
    with col4:
        success_rate = stats.get('success_rate', 0)
        st.metric("Success Rate", f"{float(success_rate):.1%}")        
    with col5:
        avg_cert = stats.get('avg_ai_certainty', 0)
        st.metric("Avg AI Certainty", f"{float(avg_cert):.1%}")
        

    # --- DATA VISUALIZATION ---
    st.subheader("Data Proportions")
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.write("**Category Distribution**")        
        # Excludes 0s.
        cat_counts = filtered_df[MASTER_COLUMNS[0]].value_counts()
        
        if not cat_counts.empty:
            fig_cat = px.pie(
                values=cat_counts.values, 
                names=cat_counts.index, 
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_cat.update_traces(textinfo='label+percent')
            st.plotly_chart(fig_cat, width='stretch')
        else:
            st.info("No categories to display.")

    with chart_col2:
        st.write("**Priority Distribution**")
        # Excludes 0s.
        prio_counts = filtered_df[MASTER_COLUMNS[1]].value_counts()
        
        if not prio_counts.empty:
            fig_prio = px.pie(
                values=prio_counts.values, 
                names=prio_counts.index, 
                hole=0.4,
                color=prio_counts.index,
                color_discrete_map={'High':'#EF553B', 'Medium':'#FECB52', 'Low':'#00CC96'}
            )
            fig_prio.update_traces(textinfo='label+percent')
            st.plotly_chart(fig_prio, width='stretch')
        else:
            st.info("No priorities to display.")

    # --- DATA TABLES ---
    st.subheader("Detailed Ticket Log")
    if filtered_df.empty:
        st.info("No tickets match the current filters.")
    else:
        st.dataframe(filtered_df, width='stretch', hide_index=True)

    # --- SAFETY NET ---
    # Identifies tickets where the AI failed OR lacked confidence
    urgent_df = df[(df[MASTER_COLUMNS[1]] == MASTER_COLUMNS_FALLBACK[1]) | (df[MASTER_COLUMNS[2]] < CONFIDENCE_THRESHOLD)]
    if not urgent_df.empty:
        st.error(f"🚨 Action Required - {len(urgent_df)} tickets failed automated classification or fell below the {CONFIDENCE_THRESHOLD:.0%} certainty threshold.")
        st.dataframe(urgent_df, width='stretch', hide_index=True)

else:
    st.warning("No data found. Please run the classifier and analyst to generate data for this dashboard.")