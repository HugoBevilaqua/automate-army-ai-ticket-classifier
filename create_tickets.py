import os
import json
import csv
import pandas as pd
from google import genai
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed
from config import TICKET_CREATOR_COLUMNS, MODEL_NAME, TEST_FILE, OUTPUT_DIR

# Structured output schema (JSON enforcement)
# We define an array-based JSON schema to ensure the AI returns a structured 
# list of objects that perfectly match our project's desired CSV headers.
ai_columns = TICKET_CREATOR_COLUMNS[1:] 
batch_schema = {
    "type": "OBJECT",
    "properties": {
        "tickets": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {col: {"type": "STRING"} for col in ai_columns},
                "required": ai_columns
            }
        }
    },
    "required": ["tickets"]
}

# Prompting
# We provide the AI with context to ensure the 
# generated tickets are realistic and diverse.
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))   
def request_batch(client, count): 
    return client.models.generate_content(
        model=MODEL_NAME,
        contents=(
            f"Generate {count} unique customer support tickets for a SaaS project management company. "
            "Use a mix of common and diverse human names. Issues should be things such as realistic software bugs, "
            "billing questions, account problems, feedback, or feature requests. Do not repeat names or issues. "
            "These are meant to be classified by another system, so DO NOT label them (e.g 'URGENT: I have a problem' or 'FEEDBACK: Please add a button')."
        ),
        config={
            "response_mime_type": "application/json",
            "response_schema": batch_schema
        }
    )

# Utility script to generate synthetic support tickets for testing and demo purposes.
def generate_test_data(client):
    print("--- Dynamic Ticket Generator ---")
    # User guardrail
    # Prevent accidental high-cost API usage by requiring confirmation for large batches.
    while True:
        try:
            count = int(input("How many tickets would you like to generate? ") or 10)
        except ValueError:
            print("Invalid input. Please enter a valid number.")
            continue
        
        if count > 100:
            print(f"\nWARNING: You are requesting {count} tickets.")
            print("Large requests may take significant time and API tokens.")
            confirm = input("Are you REALLY sure you want to proceed? (yes/no): ").lower().strip()

            if confirm == 'no':
                print("Resetting... Please enter a smaller number.")
                continue
        
        break

    filename = input("Enter filename, or leave blank for default: ").strip()
    if not filename:
        full_path = TEST_FILE  # Use the default from config.py
    else:
        # Sanitize the custom name
        filename = os.path.basename(filename)
        if not filename.endswith(".csv"): 
            filename += ".csv"
        full_path = os.path.join(OUTPUT_DIR, filename)


    print(f"Requesting a batch of {count} unique tickets...")
    
    # Tagging
    # After receiving the AI text, we programmatically add a 'ticket_id'.
    response = request_batch(client, count)
    raw_data = json.loads(response.text)
    ticket_list = raw_data["tickets"]
    for i, ticket in enumerate(ticket_list):
        ticket["ticket_id"] = f"TKT-{1000 + i}"

    # Export
    # We use QUOTE_ALL to ensure that even if a generated ticket contains 
    # commas or special characters, the CSV structure remains intact.
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    df = pd.DataFrame(ticket_list)[TICKET_CREATOR_COLUMNS]
    df.to_csv(full_path, index=False, quoting=csv.QUOTE_ALL, quotechar='"')
    print(f"--- DONE! Check '{full_path}' ---")
    input("Press Enter to close...")

# Standalone execution block for testing/debugging
if __name__ == "__main__":
    load_dotenv(override=True)
    api_key = os.getenv("PROJECT_API_KEY")
    if not api_key:
        raise ValueError("PROJECT_API_KEY not found. Check your .env file!")
    standalone_client = genai.Client(api_key=api_key)
    generate_test_data(standalone_client)