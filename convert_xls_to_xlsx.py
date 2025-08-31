import pandas as pd

# Input and output file
input_file = "Training.xls"
output_file = "Training_converted.xlsx"

try:
    # Read the Excel file (works for both .xls and .xlsx)
    df = pd.read_excel(input_file, engine="xlrd")  # xlrd only needed if real .xls
except:
    # If it fails, try openpyxl (for .xlsx disguised as .xls)
    df = pd.read_excel(input_file, engine="openpyxl")

# Save to proper .xlsx format
df.to_excel(output_file, index=False, engine="openpyxl")

print(f"✅ Converted '{input_file}' to '{output_file}' successfully!")