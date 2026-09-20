
import os
import pandas as pd


def combineBehaves(directory):
    # Ensure directory exists
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return

    # Get list of all Excel files in the directory
    excel_files = [f for f in os.listdir(directory) if f.endswith(('.xlsx', '.xls'))]
    if not excel_files:
        print("No Excel files found in the directory.")
        return

    # Read and combine only rows where State == "feed"
    combined_df = pd.DataFrame()
    for file in excel_files:
        file_path = os.path.join(directory, file)
        try:
            df = pd.read_excel(file_path)

            if 'State' not in df.columns:
                print(f"Skipping {file} — no 'State' column found.")
                continue

            feed_rows = df[df['State'].astype(str).str.lower() == 'feed']
            feed_rows['source_file'] = file  # Optional: keep track of origin
            combined_df = pd.concat([combined_df, feed_rows], ignore_index=True)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    # Save the combined dataframe
    output_file = os.path.join(directory, 'behaves_MASTER_feedonly.xlsx')
    combined_df.to_excel(output_file, index=False)
    print(f"Saved filtered Excel to: {output_file}")

combineBehaves(r"C:\\_temp workspace\\1ER-LAT-FEED\\data\\tags\\behaves")
