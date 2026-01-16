import os
import sys
import subprocess
from classifier import run_classification
from analyst import compute_stats
from csv_handler import get_ticket_column
from google import genai
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed
from config import INPUT_FILE, OUTPUT_FILE, MODEL_NAME

def main():    
    print("---------------------------")
    print("Support Ticket Classifier")
    print("---------------------------")

    # 1. Environment Setup
    # Load the sensitive API key using dotenv to keep it out of the code. 
    load_dotenv(override=True)
    api_key = os.getenv("PROJECT_API_KEY")
    if not api_key:
        raise ValueError("PROJECT_API_KEY not found. Check your .env file!")
    # Setup the Client
    client = genai.Client(api_key=api_key)


    # System warm-up (Handshake test)
    # Ensure the API key is valid and the model is reachable 
    # before the user starts the data intake process.
    print("Perfoming warm-up...")
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def handshake():    
        return client.models.generate_content(
            model=MODEL_NAME,
            contents="ping"
        )
    test_response = handshake()
    actual_model = test_response.model_version
    print(f"Connection success.")
    print(f"Model Engine: {actual_model}")
    
    # 3. Data intake
    # Allows the user to specify a file or default to the config path.
    file_path = input("Enter the path to your tickets.csv (or leave blank to use the default file): ").strip()
    if not file_path:
        file_path = INPUT_FILE
    
    if not os.path.exists(file_path):
        print(f"Error: Could not find {file_path}")
        return
    
    print(f"Loading {file_path}...")

    # 4. Data handling (csv_handler)
    # We use Gemini to intelligently identify which column contains the ticket description text 
    # and whether the file contains headers, reducing manual configuration for different CSV structures.
    # This implements a 'Human-in-the-loop' design where the AI suggests the configuration, 
    # but the user provides the final confirmation.
    try:
        target_col, df = get_ticket_column(file_path, client)
    except Exception as e:
        print(f"Handling failed: {e}")
        return

    # 5. Classification (classifier)
    # The core engine: forwards ticket contents to Gemini for category and priority labels, 
    # as well as certainty scores and reasoning.
    try:
        run_classification(df, client, target_col)
    except Exception as e:
        print(f"Classification failed: {e}")
        return
    
    # 6. Statistics (analyst)
    # Generates a summary CSV used by the dashboard to display insights.
    try:
        compute_stats(OUTPUT_FILE)
    except Exception as e:
        print(f"Analysis failed: {e}")
        return

    # 7. Visualization (dashboard)
    # Launch the Streamlit UI as a subprocess to display stats.
    print("\nLaunching Dashboard...")
    # This runs 'streamlit run dashboard.py' as a separate process
    subprocess.run(["streamlit", "run", "dashboard.py"])

if __name__ == "__main__":
    try:
        main()
    # Prevents messy tracebacks when manually interrupting the program with Ctrl+C
    except KeyboardInterrupt:
        print("\nShutting Down...")
        sys.exit(0)
    # Prints any problems
    except Exception as e:
        print(f"Oops! Something went wrong:\n {e}")
        input("Press Enter to close...")
