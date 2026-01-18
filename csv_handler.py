import os
import pandas as pd
from google import genai
from google.genai import types
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed
from config import INPUT_FILE, MODEL_NAME

# Uses Gemini to determine whether there are headers in the CSV file
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def check_for_headers(client: genai.Client, row_1: list, row_2: list) -> str:
    """Use Gemini to determine if the first row contains headers or data."""
    print("Gemini is checking for headers...")
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=(
            f"Analyze these two rows from a CSV file meant to store Customer Support Tickets.\n\n"
            f"Row 1: {row_1}\n"
            f"Row 2: {row_2}\n\n"
            f"Your task is to determine whether this CSV file's first row contains headers."
            f"Answer 'HEADERS' if Row 1 contains field labels. Examples: 'ticket_id', 'date', 'customer', 'summary', 'priority'. It's also possible for them to be labeled '0', '1', '2', etc."
            f"Answer 'DATA' if Row 1 contains actual support information. Examples: '2024-05-10', 'John Doe', 'Login error', 'Urgent', 'Priority', etc."
            f"Compare Row 1 to Row 2. If Row 1 is structurally different from Row 2 (labels vs values), it is 'HEADERS'. But if Row 1 and Row 2 contain the same type of information, Row 1 is 'DATA'."
        ),
        config=types.GenerateContentConfig(
            response_mime_type="text/x.enum",
            response_schema={
                "type": "STRING",
                "enum": ["HEADERS", "DATA"]
            }
        )
    )
    return response.text.strip()

# Uses Gemini to pick the most relevant column for ticket descriptions
@retry(stop=stop_after_attempt(3), wait=wait_fixed(10))
def pick_header(client: genai.Client, headers: list, sample: list) -> str:
    """Use Gemini to identify which column contains ticket descriptions."""
    print("Gemini is identifying the ticket description column...")
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=(
            f"Which of these columns contains the main text or description of a support ticket?\n\n"
            f"COLUMN LABELS: {headers}\n"
            f"SAMPLE FROM FIRST ROW: {sample}"            
        ),
        config=types.GenerateContentConfig(
            response_mime_type="text/x.enum",
            response_schema={
            "type": "STRING",
            "enum": headers  # Forces Gemini to pick one of the headers
            }
        )
    )
    return response.text.strip()

# Main intake workflow to load CSV and identify target column
def get_ticket_column(file_path: str, client: genai.Client) -> tuple[str, pd.DataFrame]:
    """Load CSV and identify target column for ticket descriptions."""
    # 1. Pre-screening (Peeking)
    # We only read the first 2 rows to minimize memory usage and API token costs.
    # This 'peek' allows Gemini to compare Row 1 (potential headers) with Row 2 (actual data).
    peek_df = pd.read_csv(file_path, header=None, nrows=2, skip_blank_lines=True)
    row_1 = peek_df.iloc[0].tolist()
    row_2 = peek_df.iloc[1].tolist() if len(peek_df) > 1 else "N/A (Single row file)"

    # 2. Structural reasoning
    # We use LLM reasoning to determine if the first row is 'Headers' or 'Data'.
    # By showing it the first two rows we allow it to check for contrast.
    try:
        has_header = check_for_headers(client, row_1, row_2)
    except Exception as e:
        print(f"Gemini could not determine header status: {e}")
        has_header = "HEADERS" # Fall back to 'HEADERS' mode by default
    
    # 3. Human-in-the-loop confirmation 1
    # The AI suggests, but the human confirms or overrides.
    row_1 = [str(col) if pd.notna(col) and col != "" else "EMPTY" for col in row_1] # Replaces empty cells with "Empty" to polish the printed text
    print(f"First row contains: {row_1}")
    if has_header == "HEADERS":
        print("Gemini thinks this file has headers.")
    else:
        print("Gemini thinks this file doesn't have headers.")
    header_confirm = input("Is this correct? (y/n): ").strip().lower()
    if header_confirm == 'n':
        has_header = "DATA" if has_header == "HEADERS" else "HEADERS"
        print(f"Switched mode.")

    # 4. Data loading and formatting
    # If headers weren't found, assign generic labels
    if has_header == "HEADERS":
        print("Headers detected.")
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
    else:
        print("No headers detected. Assigning generic names...")
        df = pd.read_csv(file_path, header=None)
        df.columns = [f"column_{i}" for i in range(len(df.columns))]
    
    print("--- DONE! ---")

    headers = df.columns.tolist()
    sample = df.iloc[0].tolist()

    # 5. Target identification
    # Instead of hardcoding 'description', we ask Gemini to find the most relevant column.
    # We use response_schema 'enum' to force the AI to choose a valid existing column.
    try:
        ai_suggestion = pick_header(client, headers, sample)
        ai_index = headers.index(ai_suggestion)
    except:
        print("Gemini failed to find identify target column")
        ai_suggestion = None
        ai_index = "N/A"

    # 6. Human-in-the-loop confirmation 2
    # The AI suggests, but the human confirms or overrides.
    print("\n--- Available Columns ---")
    for i, col in enumerate(headers):
        print(f"[{i}] {col}")

    if ai_suggestion is not None:
        print(f"\nGemini identified column [{ai_index}]: '{ai_suggestion}' as target column for ticket descriptions")
        user_input = input("Press ENTER to confirm, or type the INDEX number to change: ").strip()
    else:
        user_input = input("Type the INDEX number of the description column: ").strip()

    # Resolve selection: User Index -> AI Suggestion -> Error
    if user_input.isdigit() and int(user_input) < len(headers):
        target_col = headers[int(user_input)]
    else:
        target_col = ai_suggestion

    if not target_col:
        raise LookupError("Could not determine a target column for ticket descriptions.")

    return target_col, df

# Standalone execution block for testing/debugging
if __name__ == "__main__":
    load_dotenv(override=True)
    api_key = os.getenv("PROJECT_API_KEY")
    if not api_key:
        raise ValueError("PROJECT_API_KEY not found. Check your .env file!")
    standalone_client = genai.Client(api_key=api_key)
    target, dataframe = get_ticket_column(INPUT_FILE, standalone_client)
    print("\n--- TEST RUN RESULTS ---")
    print(f"Final Target Column: {target}")
    print(f"DataFrame loaded successfully with {len(dataframe)} rows.")
    print("First row of data:")
    print(dataframe.iloc[0])