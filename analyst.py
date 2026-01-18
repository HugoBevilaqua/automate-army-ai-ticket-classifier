import os
import pandas as pd
from config import CONFIDENCE_THRESHOLD, MASTER_CATEGORIES, MASTER_PRIORITIES, MASTER_COLUMNS, MASTER_COLUMNS_FALLBACK, STATS_FILE, OUTPUT_FILE

# Reads the enriched CSV and generates an aggregation summary.
def compute_stats(file_path: str) -> None: 
    """Compute aggregated statistics from enriched tickets and save to CSV."""   
    print("\nAnalyzing results to generate stats...")
    # Data loading
    df = pd.read_csv(file_path)
    # Total number of tickets
    total = len(df)
    # Failed classifications
    errors = df[df[MASTER_COLUMNS[0]] == MASTER_COLUMNS_FALLBACK[0]]
    # Successful classifications
    success = df[df[MASTER_COLUMNS[0]] != MASTER_COLUMNS_FALLBACK[0]]
    # Successful classifications under the certainty threshold
    low_cert_success = success[success[MASTER_COLUMNS[2]] < CONFIDENCE_THRESHOLD]
    # Successful classifications at or above the certainty threshold
    high_cert_success = success[success[MASTER_COLUMNS[2]] >= CONFIDENCE_THRESHOLD]
    # Number of failed classifications
    num_errors = len(errors)
    # Number of successful classifications
    num_success = len(success)
    # Number of successful classifications under the certainty threshold
    num_low_cert = len(low_cert_success)
    # Number of successful classifications at or above the certainty threshold
    num_high_cert = len(high_cert_success)
    # Number of tickets that require review
    num_manual = num_errors + num_low_cert
    # Average certainty score
    avg_cert = success[MASTER_COLUMNS[2]].mean() if num_success > 0 else 0

    # Building the stats list
    report = [
        ["total_processed", total],
        ["success_count", num_success],
        ["error_count", num_errors],
        ["low_certainty_count", num_low_cert],
        ["high_certainty_count", num_high_cert],
        ["review_required_count", num_manual],
        ["success_rate", (num_success / total) if total > 0 else 0.0],
        ["manual_review_rate", (num_manual / total) if total > 0 else 0.0],
        ["avg_ai_certainty", avg_cert]
    ]

    # Adding category counts
    for category in MASTER_CATEGORIES:
        # Filter the success_df for this specific category and count rows
        count = len(success[success[MASTER_COLUMNS[0]] == category])
        # Append as [metric_name, value]
        report.append([f"{MASTER_COLUMNS[0]}_{category.lower()}", count])

    # Adding priority counts
    for priority in MASTER_PRIORITIES:
        # Filter the success_df for this specific priority and count rows
        count = len(success[success[MASTER_COLUMNS[1]] == priority])
        # Append as [metric_name, value]
        report.append([f"{MASTER_COLUMNS[1]}_{priority.lower()}", count])

    # Create dataframe and save .csv file
    stats_df = pd.DataFrame(report, columns=["metric", "value"])
    os.makedirs(os.path.dirname(STATS_FILE), exist_ok=True)
    stats_df.to_csv(STATS_FILE, index=False)

    print(f"--- DONE! Check '{STATS_FILE}' ---")

# Standalone execution block for testing/debugging
if __name__ == "__main__":
    compute_stats(OUTPUT_FILE)