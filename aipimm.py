import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="AI Attribute Mapper 2.2", layout="wide")
st.title("AI Attribute Mapper 2.3")

MEMORY_FILE = 'mappings_memory.csv'
VALUE_MAP_FILE = 'value_mapped_results.csv'

def load_excel(uploaded_file):
    return pd.read_excel(uploaded_file, sheet_name=None, engine='openpyxl')

def load_memory(path, columns):
    if not os.path.exists(path):
        return pd.DataFrame(columns=columns)
    try:
        df = pd.read_csv(path)
        for col in columns:
            if col not in df.columns:
                df[col] = ''
        return df
    except:
        return pd.DataFrame(columns=columns)

def save_memory(path, df):
    df.to_csv(path, index=False)

# Upload Excel
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])
if uploaded_file:
    sheets = load_excel(uploaded_file)
    st.session_state['sheets'] = sheets
elif 'sheets' in st.session_state:
    sheets = st.session_state['sheets']
else:
    st.stop()

# Select sheets
st.write("Sheets found:", list(sheets.keys()))
selected_markets = st.multiselect("Select Marketplace Sheets", [s for s in sheets if s.lower() != "global"])
if not selected_markets:
    st.stop()

# Parse all attributes and values
all_rows = []
for sheet_name in selected_markets:
    df = sheets[sheet_name]
    if df.empty or df.shape[1] < 2:
        continue
    for _, row in df.iterrows():
        for i in range(1, len(row)):
            attr = str(df.columns[i]).strip()
            val_str = str(row.iloc[i]) if pd.notna(row.iloc[i]) else ""
            all_rows.append({
                "Marketplace": sheet_name,
                "Marketplace Attribute": attr,
                "Marketplace Value": val_str.strip()
            })

value_df = pd.DataFrame(all_rows)

# Load attribute mapping memory
memory_df = load_memory(MEMORY_FILE, ['Marketplace Attribute', 'Global Attribute'])
memory_lookup = dict(zip(memory_df['Marketplace Attribute'], memory_df['Global Attribute']))
value_df['Global Attribute'] = value_df['Marketplace Attribute'].map(memory_lookup).fillna(value_df['Marketplace Attribute'])

# Global Value = Marketplace Value
value_df['Global Value'] = value_df['Marketplace Value']

# Sort for sequencing
value_df = value_df.sort_values(by=['Global Attribute', 'Marketplace'])

# Show and export
st.subheader("✅ Final Sequenced Output by Global Attribute")
st.dataframe(value_df)

csv_data = value_df.to_csv(index=False).encode('utf-8')
st.download_button("Download Sequenced Mapping", data=csv_data, file_name="final_sequenced_mapping.csv")

save_memory(VALUE_MAP_FILE, value_df)
