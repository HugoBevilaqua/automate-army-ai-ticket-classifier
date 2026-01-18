import os
import json
import time
import csv
import pandas as pd
from google import genai
from google.genai import errors
from google.genai import types
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt
from config import MASTER_COLUMNS, MASTER_COLUMNS_FALLBACK, MODEL_NAME, OUTPUT_FILE, INPUT_FILE, CLASSIFICATION_SCHEMA, WAIT_TIME

# 1. Observability (Logging)
# We define a custom logger to provide clear terminal feedback during retries.
# This distinguishes between 'Server Overload' and 'Rate Limits', helping
# the user understand the cause of a failed attempt.
def log_retry(retry_state):
    """Log retry attempts with user-friendly error messages."""
    # retry_state.outcome.exception() gives us the actual error object
    err = retry_state.outcome.exception()        
    # We simplify the message for the terminal
    if isinstance(err, errors.ServerError):
        msg = "Server busy (503 Overloaded)"
    elif "429" in str(err):
        msg = "Rate limit reached (429)"
    else:
        msg = str(err).split('.')[0] # Get just the first part of the error            
    print(f"  Attempt {retry_state.attempt_number} failed: {msg}. Retrying...")

# 2. Resilience logic (Exponential backoff)
# Cloud APIs can be unstable. We use wait_exponential to increase the wait time between retries, 
# giving the server time to recover while preventing our script from crashing.
@retry(
    wait=wait_exponential(multiplier=2, min=2, max=30), 
    stop=stop_after_attempt(10),
    after=log_retry, # This tells Tenacity to run our log function
    reraise=True
)
def classify_ticket(client: genai.Client, description: str) -> dict:
    """Classify a support ticket using Gemini with exponential backoff retry logic."""
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=f"Classify this support ticket: '{description}'",
        config=types.GenerateContentConfig(
            response_mime_type='application/json',
            response_schema=CLASSIFICATION_SCHEMA
        )
    )
    return json.loads(response.text)

# Main classification workflow to enrich the DataFrame with AI-generated labels
def run_classification(df: pd.DataFrame, client: genai.Client, target_col: str, progress_callback: callable | None = None) -> None:
    """Classify all tickets in a DataFrame and save enriched results to CSV."""
    # 3. Iterative processing
    total_tickets = len(df)
    # This creates: {"col1": [], "col2": [], ...} dynamically
    result_dict = {col: [] for col in MASTER_COLUMNS}

    print(f"\nClassifying {total_tickets} tickets with Gemini...")

    # Iterate through each row in the enumerated DataFrame
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        try:
            # Get our classifications for each row
            print(f"[{i}/{total_tickets}] Processing ticket...")
            if progress_callback:
                progress_callback(i, total_tickets, "processing")
            
            result = classify_ticket(client, row[target_col])
            for idx, col in enumerate(MASTER_COLUMNS):
                # We use .get() with a fallback to ensure the script continues 
                # even if one JSON field is missing.
                value = result.get(col, MASTER_COLUMNS_FALLBACK[idx])
                result_dict[col].append(value)
            # API courtesy: Add a "breather" between tickets to stay under the radar
            time.sleep(WAIT_TIME) 
        except Exception as e:
            print(f"Failed to classify ticket {i} after maximum retries.")
            for idx, col in enumerate(MASTER_COLUMNS):
                val = MASTER_COLUMNS_FALLBACK[idx] if idx < 3 else f"Error: {str(e)[:40]}"
                result_dict[col].append(val)

    # 4. Data Integrity (Legacy Protection)
    # If the output file already contains our target columns, we back up the 
    # original data by renaming it (e.g., 'Category' becomes 'Category_original').
    # This prevents destructive overwrites of existing human or AI work.
    for col in MASTER_COLUMNS:
        if col in df.columns:
            legacy_name = f"{col}_original"
            counter = 2
            # Keep counting up until a name is available
            while legacy_name in df.columns:
                legacy_name = f"{col}_original_{counter}"
                counter += 1
            print(f"Column '{col}' already exists. Renaming existing data to '{legacy_name}'...")
            df.rename(columns={col: legacy_name}, inplace=True)
    
    for col in MASTER_COLUMNS:
        df[col] = result_dict[col]

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, quoting=csv.QUOTE_NONNUMERIC)
    
    if progress_callback:
        progress_callback(total_tickets, total_tickets, "complete")
    
    print(f"--- DONE! Check '{OUTPUT_FILE}' ---")

# Standalone execution block for testing/debugging
if __name__ == "__main__":
    load_dotenv(override=True)
    api_key = os.getenv("PROJECT_API_KEY")
    if not api_key:
        raise ValueError("PROJECT_API_KEY not found. Check your .env file!")
    standalone_client = genai.Client(api_key=api_key)

    test_df = pd.read_csv(INPUT_FILE)
    run_classification(test_df, standalone_client, "issue_description") # Change 'issue_description' to the target column of the .csv you want to test