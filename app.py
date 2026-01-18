import os
import io
import csv
import pandas as pd
import streamlit as st
import plotly.express as px
from google import genai
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed
from csv_handler import check_for_headers, pick_header
from classifier import run_classification
from analyst import compute_stats
from config import MODEL_NAME, OUTPUT_FILE, MASTER_CATEGORIES, MASTER_PRIORITIES, MASTER_COLUMNS, MASTER_COLUMNS_FALLBACK, CONFIDENCE_THRESHOLD, STATS_FILE

# Page configuration
st.set_page_config(page_title="Ticket Classifier UI", page_icon="🎫", layout="wide")

# Initialize session state.
# Session state variables are special streamlit containers that persist across user interactions,
# allowing us to maintain context in this multi-step application. 
if "stage" not in st.session_state:
    st.session_state.stage = "warmup"
if "client" not in st.session_state:
    st.session_state.client = None
if "uploaded_file_bytes" not in st.session_state:
    st.session_state.uploaded_file_bytes = None
if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None
if "headers_detected" not in st.session_state:
    st.session_state.headers_detected = None
if "target_col" not in st.session_state:
    st.session_state.target_col = None
if "column_options" not in st.session_state:
    st.session_state.column_options = None
if "gemini_suggestion" not in st.session_state:
    st.session_state.gemini_suggestion = None
if "col_confirm_area" not in st.session_state:
    st.session_state.col_confirm_area = None

# --- Dynamic Title and Subtitle ---
def display_title_and_subtitle():
    """Display dynamic title and subtitle based on stage."""
    if st.session_state.stage == "complete":
        st.title("🎫 Support Ticket Dashboard")
        st.markdown("Analysis of customer issues using Google Gemini.")
    else:
        st.title("🎫 Support Ticket Classifier")
        st.markdown("Upload your CSV file and let AI classify your support tickets.")

# --- Initialization ---
def init_client():
    """Initialize the Gemini client and test connection."""
    if st.session_state.client is not None: # Already initialized? Don't redo.
        return True
    
    load_dotenv(override=True)
    api_key = os.getenv("PROJECT_API_KEY")
    if not api_key:
        st.error("❌ PROJECT_API_KEY not found in .env file!")
        return False
    
    try:
        st.session_state.client = genai.Client(api_key=api_key)
        
        # Handshake test
        @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
        def handshake():
            return st.session_state.client.models.generate_content(
                model=MODEL_NAME,
                contents="ping"
            )
        
        response = handshake()
        st.success(f"✅ Connection successful! Model: {response.model_version}")
        return True
    except Exception as e:
        st.error(f"❌ Connection failed: {e}")
        return False

# --- Dashboard Helper Functions ---
def load_enriched_data():
    """Load the enriched tickets and statistics."""
    try:
        df = pd.read_csv(OUTPUT_FILE)
        stats_dict = {}
        try:
            stats_df = pd.read_csv(STATS_FILE)
            stats_dict = dict(zip(stats_df['metric'], stats_df['value']))
        except FileNotFoundError:
            st.warning(f"Note: '{STATS_FILE}' not found. Some metrics may be unavailable.")
        return df, stats_dict
    except FileNotFoundError:
        st.error(f"Error: '{OUTPUT_FILE}' not found.")
        return pd.DataFrame(), {}

# Display the full dashboard
def display_dashboard():
    """Display the full dashboard with metrics, charts, and data."""
    df, stats = load_enriched_data()
    
    if df.empty:
        st.warning("No enriched data available.")
        return
    

    
    # --- Sidebar Filters ---
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
    
    # Filtering logic
    filtered_df = df[
        (df[MASTER_COLUMNS[0]].isin(selected_cats)) & 
        (df[MASTER_COLUMNS[1]].isin(selected_prio))
    ]
    
    # --- Export Actions ---
    col1, col2, colspacer = st.columns([2, 2, 10])
    with col1:
        try:
            csv_data = df.to_csv(index=False, quoting=csv.QUOTE_NONNUMERIC)
            st.download_button(
                label="📥 Download Enriched Tickets",
                data=csv_data,
                file_name="enriched_tickets.csv",
                mime='text/csv',
                help="Download the complete list of tickets with AI-enriched labels"
            )
        except Exception as e:
            st.error(f"Failed to prepare download: {e}")
    
    with col2:
        try:
            stats_df = pd.read_csv(STATS_FILE)
            csv_data = stats_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Statistics",
                data=csv_data,
                file_name="statistics.csv",
                mime="text/csv",
                help="Download statistics derived from the enriched tickets."
            )
        except Exception as e:
            st.info("Statistics file not available yet.")
    
    st.divider()
    
    # --- Top-Level Metrics ---
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
    
    # --- Data Visualization (Pie Charts) ---
    st.subheader("Data Proportions")
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.write("**Category Distribution**")
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
        prio_counts = filtered_df[MASTER_COLUMNS[1]].value_counts()
        if not prio_counts.empty:
            fig_prio = px.pie(
                values=prio_counts.values,
                names=prio_counts.index,
                hole=0.4,
                color=prio_counts.index,
                color_discrete_map={'High': '#EF553B', 'Medium': '#FECB52', 'Low': '#00CC96'}
            )
            fig_prio.update_traces(textinfo='label+percent')
            st.plotly_chart(fig_prio, width='stretch')
        else:
            st.info("No priorities to display.")
    
    # --- Data Tables ---
    st.subheader("Detailed Ticket Log")
    if filtered_df.empty:
        st.info("No tickets match the current filters.")
    else:
        st.dataframe(filtered_df, width='stretch', hide_index=True)
    
    # --- Safety Net ---
    urgent_df = df[(df[MASTER_COLUMNS[1]] == MASTER_COLUMNS_FALLBACK[1]) | (df[MASTER_COLUMNS[2]] < CONFIDENCE_THRESHOLD)]
    if not urgent_df.empty:
        st.error(f"🚨 Action Required - {len(urgent_df)} tickets failed automated classification or fell below the {CONFIDENCE_THRESHOLD:.0%} certainty threshold.")
        st.dataframe(urgent_df, width='stretch', hide_index=True)

# --- Wrapper for Classification with Progress ---
def run_classification_with_progress(df, client, target_col):
    """Wrapper that provides Streamlit progress updates for run_classification."""
    total_tickets = len(df)
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    def update_progress(current, total, status):
        """Callback function to update Streamlit UI during classification."""
        progress = current / total
        progress_bar.progress(progress)
        if status == "complete":
            progress_text.text("✅ Classification complete!")
        else:
            progress_text.text(f"[{current}/{total}]")
    
    # Call the original classifier with our progress callback
    run_classification(df, client, target_col, progress_callback=update_progress) # We call run_classification with an added argument to update progress.
    
    st.success(f"Successfully processed {total_tickets} tickets!")

# --- Main UI ---
display_title_and_subtitle()

# Persistent container for Step 3 UI
if st.session_state.col_confirm_area is None:
    st.session_state.col_confirm_area = st.empty()

st.divider()

# Stage 0: Warmup
if st.session_state.stage == "warmup":
    st.subheader("Step 0: Warming Up")
    st.write("Initializing Gemini connection...")
    
    if init_client():
        import time
        time.sleep(1)  # Let the success message display
        st.session_state.stage = "upload"
        st.rerun()
    else:
        st.stop()

# Stage 1: File Upload
elif st.session_state.stage == "upload":
    st.subheader("Step 1: Upload Your CSV File")
    
    # File uploader (removed init_client call - already done in warmup)
    uploaded_file = st.file_uploader(
        "Drag and drop your CSV file here or click to browse",
        type=["csv"],
        help="Upload a CSV file containing your support tickets"
    )
    
    if uploaded_file is not None:
        try:
            # Store the raw file bytes for later processing
            file_bytes = uploaded_file.getvalue()
            st.session_state.uploaded_file_bytes = file_bytes
            
            # Read WITHOUT assuming headers for preview
            df_preview = pd.read_csv(io.BytesIO(file_bytes), header=None, nrows=5)
            
            st.success(f"✅ File loaded! Found {uploaded_file.name}")
            st.write("**Preview of first 5 rows (raw, no header assumption):**")
            st.dataframe(df_preview, width='stretch')
            
            # Proceed button
            if st.button("Proceed to Header Detection", type="primary"):
                st.session_state.stage = "header_confirm"
                st.rerun()
        except Exception as e:
            st.error(f"❌ Failed to read file: {e}")

# Stage 2: Header Confirmation
elif st.session_state.stage == "header_confirm":
    st.subheader("Step 2: Confirm Header Detection")
    
    if st.session_state.uploaded_file_bytes is None:
        st.error("No file loaded. Going back...")
        st.session_state.stage = "upload"
        st.rerun()
    
    # Analyze headers with Gemini
    if st.session_state.headers_detected is None:
        with st.spinner("🔍 Analyzing file structure..."):
            try:
                # Read file without headers to get raw rows
                peek_df = pd.read_csv(io.BytesIO(st.session_state.uploaded_file_bytes), header=None, nrows=2)
                row_1 = peek_df.iloc[0].tolist()
                row_2 = peek_df.iloc[1].tolist() if len(peek_df) > 1 else "N/A (Single row file)"
                
                # Import from csv_handler instead of defining here
                has_header = check_for_headers(st.session_state.client, row_1, row_2)
                st.session_state.headers_detected = has_header
                st.session_state.gemini_suggestion = has_header
            except Exception as e:
                st.error(f"❌ Failed to detect headers: {e}")
                st.session_state.headers_detected = "HEADERS"
                st.session_state.gemini_suggestion = "HEADERS"
    
    # Display detection result - always show Gemini's original suggestion
    st.info(f"**Gemini's Analysis:** The first row appears to have **{st.session_state.gemini_suggestion}**")
    
    # Show the raw first row
    peek_df = pd.read_csv(io.BytesIO(st.session_state.uploaded_file_bytes), header=None, nrows=1)
    st.write("**First row contains:**")
    st.dataframe(peek_df, width='stretch', hide_index=True)
    
    # Confirmation buttons - Dynamic based on Gemini's suggestion
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    if st.session_state.gemini_suggestion == "HEADERS": # Headers detected
        with col1:
            if st.button("✅ Correct, has headers", type="primary", width='stretch'):
                st.session_state.headers_detected = "HEADERS"
                st.session_state.uploaded_df = pd.read_csv(io.BytesIO(st.session_state.uploaded_file_bytes))
                st.session_state.uploaded_df.columns = st.session_state.uploaded_df.columns.str.strip()
                st.session_state.stage = "col_confirm"
                st.rerun()
        with col2:
            if st.button("❌ Incorrect, no headers", width='stretch'):
                st.session_state.headers_detected = "DATA"
                st.session_state.uploaded_df = pd.read_csv(io.BytesIO(st.session_state.uploaded_file_bytes), header=None)
                st.session_state.uploaded_df.columns = [f"column_{i}" for i in range(len(st.session_state.uploaded_df.columns))]
                st.session_state.stage = "col_confirm"
                st.rerun()
    else:  # Data detected
        with col1:
            if st.button("✅ Correct, no headers", type="primary", width='stretch'):
                st.session_state.headers_detected = "DATA"
                st.session_state.uploaded_df = pd.read_csv(io.BytesIO(st.session_state.uploaded_file_bytes), header=None)
                st.session_state.uploaded_df.columns = [f"column_{i}" for i in range(len(st.session_state.uploaded_df.columns))]
                st.session_state.stage = "col_confirm"
                st.rerun()
        with col2:
            if st.button("❌ Incorrect, has headers", width='stretch'):
                st.session_state.headers_detected = "HEADERS"
                st.session_state.uploaded_df = pd.read_csv(io.BytesIO(st.session_state.uploaded_file_bytes))
                st.session_state.uploaded_df.columns = st.session_state.uploaded_df.columns.str.strip()
                st.session_state.stage = "col_confirm"
                st.rerun()
    
    with col3:
        if st.button("🔄 Back", width='stretch'):
            st.session_state.stage = "upload"
            st.session_state.uploaded_file_bytes = None
            st.session_state.headers_detected = None
            st.session_state.gemini_suggestion = None
            st.rerun()

# Stage 3: Column Selection
elif st.session_state.stage == "col_confirm":
    with st.session_state.col_confirm_area.container():
        st.divider()
        st.subheader("Step 3: Select Description Column")
        
        if st.session_state.uploaded_df is None:
            st.error("No file loaded. Going back...")
            st.session_state.stage = "upload"
            st.rerun()
        
        df = st.session_state.uploaded_df.copy()
        headers = df.columns.tolist()
        sample = df.iloc[0].tolist()
        
        # Identify target column with Gemini
        if st.session_state.column_options is None:
            with st.spinner("🔍 Analyzing columns..."):
                try:
                    headers = df.columns.tolist()
                    sample = df.iloc[0].tolist()
                    
                    # Import from csv_handler instead of defining here
                    ai_suggestion = pick_header(st.session_state.client, headers, sample)
                    st.session_state.column_options = {"headers": headers, "ai_suggestion": ai_suggestion}
                except Exception as e:
                    st.warning(f"⚠️ Failed to auto-detect column: {e}")
                    st.session_state.column_options = {"headers": headers, "ai_suggestion": None}
        
        headers = st.session_state.column_options["headers"]
        ai_suggestion = st.session_state.column_options["ai_suggestion"]
        
        # AI suggestion
        if ai_suggestion:
            st.success(f"✨ **Gemini suggests:** Column `{ai_suggestion}`")
        else:
            st.info("Gemini could not auto-detect the description column. Please select manually.")
        
        # User selection
        selected_idx = st.selectbox(
            "Select the column containing ticket descriptions:",
            range(len(headers)),
            format_func=lambda i: f"[{i}] {headers[i]}",
            index=headers.index(ai_suggestion) if ai_suggestion and ai_suggestion in headers else 0
        )
        
        st.session_state.target_col = headers[selected_idx]
        selected_col_name = headers[selected_idx]
        
        # Display 5-row sample of selected column
        st.subheader(f"Preview:")
        sample_data = df[[selected_col_name]].head(5)
        st.dataframe(sample_data, width='stretch', hide_index=True)
        
        st.divider()
        
        # Proceed buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Start Classification", type="primary", width='stretch'):
                st.session_state.stage = "processing"
                st.rerun()
        with col2:
            if st.button("🔄 Back", width='stretch'):
                st.session_state.stage = "header_confirm"
                st.session_state.column_options = None
                st.rerun()

# Stage 4: Processing
elif st.session_state.stage == "processing":
    # Clear Step 3 UI from screen FIRST
    if st.session_state.col_confirm_area:
        st.session_state.col_confirm_area.empty()
    
    st.divider()
    st.subheader("Step 4: Processing Tickets")
    st.write("Classifying...")
    
    if st.session_state.uploaded_df is None or st.session_state.target_col is None:
        st.error("Missing data. Going back...")
        st.session_state.stage = "upload"
        st.rerun()
    
    df = st.session_state.uploaded_df.copy()
    
    try:
        # Run classification with progress bar
        run_classification_with_progress(df, st.session_state.client, st.session_state.target_col)
        
        # Run analytics
        st.info("📈 Generating statistics...")
        compute_stats(OUTPUT_FILE)
        
        st.session_state.stage = "complete"
        st.rerun()
    except Exception as e:
        st.error(f"❌ Processing failed: {e}")
        if st.button("🔄 Try Again"):
            st.session_state.stage = "col_confirm"
            st.rerun()

# Stage 5: Complete - Full Dashboard Display
elif st.session_state.stage == "complete":
    # Display the full dashboard
    display_dashboard()
    
    st.divider()
    
    # Reset button at the bottom
    if st.button("🔄 Start Over", width='stretch'):
        st.session_state.stage = "upload"
        st.session_state.uploaded_file_bytes = None
        st.session_state.uploaded_df = None
        st.session_state.headers_detected = None
        st.session_state.target_col = None
        st.session_state.column_options = None
        st.rerun()