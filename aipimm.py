import streamlit as st
import pandas as pd
import os
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="AI Attribute Mapper", layout="wide")
st.title("AI Attribute Mapper 2.0")

MEMORY_FILE = 'mappings_memory.csv'
GLOBAL_ATTR_FILE = 'global_attributes.csv'
VALUE_MEMORY_FILE = 'value_mappings_memory.csv'
CONFIDENCE_THRESHOLD = 0.3

@st.cache_resource
def load_model():
    return SentenceTransformer('all-mpnet-base-v2')

model = load_model()

@st.cache_data
def load_excel(uploaded_file):
    return pd.read_excel(uploaded_file, sheet_name=None, engine='openpyxl')

@st.cache_data
def embed_texts(texts):
    return model.encode(texts, convert_to_tensor=True).cpu().numpy()

def generate_generic_label(attr):
    return attr.replace('_', ' ').replace('-', ' ').strip().title()

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

# Upload file
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])
if uploaded_file:
    st.session_state['sheets'] = load_excel(uploaded_file)
elif 'sheets' in st.session_state:
    sheets = st.session_state['sheets']
else:
    st.stop()

sheets = st.session_state['sheets']
st.write("Sheets found:", list(sheets.keys()))

selected_markets = st.multiselect("Select Marketplace Sheets", [s for s in sheets if s.lower() != "global"])
if not selected_markets:
    st.stop()

# Parse attribute + value pairs
global_value_map = {}
all_values = []
raw_attrs = []

for sheet_name in selected_markets:
    df = sheets[sheet_name]
    if df.empty or df.shape[1] < 2:
        continue
    df = df[df[df.columns[0]].notna()]
    for _, row in df.iterrows():
        attr = str(row.iloc[0]).strip()
        val_str = str(row.iloc[1]) if pd.notna(row.iloc[1]) else ""
        values = [v.strip() for v in val_str.split(',')] or [""]
        raw_attrs.append(attr)
        for val in values:
            global_attr = generate_generic_label(attr)
            all_values.append({
                "Marketplace": sheet_name,
                "Marketplace Attribute": attr,
                "Global Attribute": global_attr,
                "Marketplace Value": val,
                "Global Value": val  # No AI mapping for value
            })

value_df = pd.DataFrame(all_values)

# Attribute suggestion and editing
st.subheader("🔍 Suggest Global Attributes")
market_attrs = pd.Series(raw_attrs).dropna().astype(str).str.strip().tolist()

memory_df = load_memory(MEMORY_FILE, ['Marketplace Attribute', 'Global Attribute'])
memory_lookup = dict(zip(memory_df['Marketplace Attribute'], memory_df['Global Attribute']))
unique_attrs = list(dict.fromkeys(market_attrs))  # preserve order, allow duplicates

global_attr_suggestions = []
global_attr_names = list(memory_lookup.values())
attr_emb = embed_texts(unique_attrs)
global_emb = embed_texts(global_attr_names) if global_attr_names else []

for i, attr in enumerate(unique_attrs):
    if attr in memory_lookup:
        suggested = memory_lookup[attr]
        confidence = 1.0
    elif global_emb is not None and len(global_emb):
        sims = util.cos_sim(attr_emb[i], global_emb).flatten().numpy()
        best_idx = sims.argmax()
        best_score = sims[best_idx]
        suggested = global_attr_names[best_idx] if best_score >= CONFIDENCE_THRESHOLD else generate_generic_label(attr)
        confidence = round(float(best_score), 3)
    else:
        suggested = generate_generic_label(attr)
        confidence = 1.0
    global_attr_suggestions.append({
        "Marketplace Attribute": attr,
        "AI Attribute (Suggested)": suggested,
        "Mapped Attribute": suggested,
        "Confidence": confidence
    })

attr_df = pd.DataFrame(global_attr_suggestions)
st.dataframe(attr_df)

st.subheader("📝 Edit and Save Attribute Mappings")
edited_attr_df = st.data_editor(attr_df, num_rows="dynamic")
st.download_button("Download Attribute Mappings", data=edited_attr_df.to_csv(index=False).encode('utf-8'), file_name="attribute_mappings.csv")

# Save updated attribute memory
new_memory = edited_attr_df[['Marketplace Attribute', 'Mapped Attribute']].rename(columns={"Mapped Attribute": "Global Attribute"})
memory_df = pd.concat([memory_df, new_memory]).drop_duplicates('Marketplace Attribute', keep='last')
save_memory(MEMORY_FILE, memory_df)

# Save value data
save_memory("value_mapped_results.csv", value_df)

st.subheader("✅ Final Attribute + Value Mapping Output")
st.dataframe(value_df)

st.download_button("Download Final Value Mapping", data=value_df.to_csv(index=False).encode('utf-8'), file_name="final_value_mapping.csv")
