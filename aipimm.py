import streamlit as st
import pandas as pd
import os
import torch
import time
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="AI Attribute Mapper", layout="wide")
st.title("AI Attribute Mapper 3.1")

MEMORY_FILE = 'mappings_memory.csv'
VALUE_MAP_FILE = 'value_mapped_results.csv'
CONFIDENCE_THRESHOLD = 0.75

@st.cache_resource
def load_model():
    return SentenceTransformer('all-mpnet-base-v2')
model = load_model()

@st.cache_data
def load_excel(uploaded_file):
    return pd.read_excel(uploaded_file, sheet_name=None, engine='openpyxl')

def sanitize_text_input(texts):
    if isinstance(texts, (str, float, int)):
        texts = [str(texts)]
    elif isinstance(texts, list):
        texts = [str(t) for t in texts if isinstance(t, (str, float, int))]
    else:
        texts = [str(texts)]
    return texts

@st.cache_data
def embed_texts(texts):
    texts = sanitize_text_input(texts)
    return model.encode(texts, convert_to_tensor=True).cpu().numpy()

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

# Upload & load sheets
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])
if uploaded_file:
    sheets = load_excel(uploaded_file)
    st.session_state['sheets'] = sheets
elif 'sheets' in st.session_state:
    sheets = st.session_state['sheets']
else:
    st.stop()

st.write("Sheets found:", list(sheets.keys()))
selected_markets = st.multiselect("Select Marketplace Sheets", [s for s in sheets if s.lower() != "global"])
if not selected_markets:
    st.stop()

# Parse values
all_rows = []
raw_attrs = []

for sheet_name in selected_markets:
    df = sheets[sheet_name]
    if df.empty or df.shape[1] < 2:
        continue
    df = df[df[df.columns[0]].astype(str).str.strip() != ""]
    for _, row in df.iterrows():
        attr = str(row.iloc[0]).strip()
        val_str = str(row.iloc[1]) if pd.notna(row.iloc[1]) else ""
        values = [v.strip() for v in val_str.split(',')] or [""]
        raw_attrs.append(attr)
        for val in values:
            all_rows.append({
                "Marketplace": sheet_name,
                "Marketplace Attribute": attr,
                "Marketplace Value": val
            })

value_df = pd.DataFrame(all_rows)
st.subheader("🔍 Parsed Marketplace Attribute-Value Pairs")
st.dataframe(value_df)

# Attribute mapping suggestion
st.subheader("🔍 Attribute Mapping Suggestions")
memory_df = load_memory(MEMORY_FILE, ['Marketplace Attribute', 'Global Attribute'])
memory_lookup = dict(zip(memory_df['Marketplace Attribute'], memory_df['Global Attribute']))
unique_attrs = raw_attrs

global_attrs = list(set(memory_lookup.values()))
global_emb = embed_texts(global_attrs) if global_attrs else []
attr_emb = embed_texts(unique_attrs)

suggestions = []
for i, attr in enumerate(unique_attrs):
    if attr in memory_lookup:
        mapped = memory_lookup[attr]
        conf = 1.0
    elif global_emb is not None and len(global_emb):
        sims = util.cos_sim(attr_emb[i], global_emb).flatten().numpy()
        best_idx = sims.argmax()
        best_score = sims[best_idx]
        mapped = global_attrs[best_idx] if best_score >= CONFIDENCE_THRESHOLD else attr
        conf = float(best_score)
    else:
        mapped = attr
        conf = 1.0
    suggestions.append({
        "Marketplace Attribute": attr,
        "Mapped Attribute": mapped,
        "Confidence": round(conf, 3)
    })

attr_df = pd.DataFrame(suggestions)
st.dataframe(attr_df)

# Editable UI
edited_df = st.data_editor(attr_df, num_rows="dynamic")
edited_attr_map = edited_df[['Marketplace Attribute', 'Mapped Attribute']].dropna()
edited_attr_map = edited_attr_map.rename(columns={'Mapped Attribute': 'Global Attribute'})

# Save memory
memory_df = pd.concat([memory_df, edited_attr_map]).drop_duplicates('Marketplace Attribute', keep='last')
save_memory(MEMORY_FILE, memory_df)

# Merge mapping into value_df
value_df = value_df.merge(edited_attr_map, on='Marketplace Attribute', how='left')
value_df['Global Attribute'] = value_df['Global Attribute'].fillna(value_df['Marketplace Attribute'])

# Value mapping
value_df['Global Value'] = value_df['Marketplace Value']

# Final Smart Semantic Matrix Output
st.subheader("📊 Smart Semantic Matrix Output")
pivot_keys = value_df[["Global Attribute", "Global Value"]].drop_duplicates()
marketplaces = value_df["Marketplace"].unique()

# Embedding cache
embedding_cache = {}
def get_embedding(text):
    if text not in embedding_cache:
        embedding_cache[text] = model.encode(text, convert_to_tensor=True)
    return embedding_cache[text]

def semantic_match(global_val, candidate_vals):
    if global_val in candidate_vals:
        return global_val
    global_emb = get_embedding(global_val)
    candidate_embs = torch.stack([get_embedding(val) for val in candidate_vals])
    sims = util.cos_sim(global_emb, candidate_embs)[0]
    return candidate_vals[sims.argmax()]

# Timed semantic matrix construction
with st.spinner("Matching values using semantic similarity..."):
    start_time = time.time()

    semantic_rows = []
    for _, row in pivot_keys.iterrows():
        g_attr = row["Global Attribute"]
        g_val = row["Global Value"]

        result_row = {
            "PIM Attribute": g_attr,
            "PIM Attribute Value": g_val
        }

        for mp in marketplaces:
            candidates = value_df[
                (value_df["Global Attribute"] == g_attr) &
                (value_df["Marketplace"] == mp)
            ]["Marketplace Value"].tolist()

            result_row[mp] = semantic_match(g_val, candidates) if candidates else "N/A"

        semantic_rows.append(result_row)

    semantic_df = pd.DataFrame(semantic_rows).sort_values(by=["PIM Attribute", "PIM Attribute Value"]).reset_index(drop=True)
    st.success(f"Semantic mapping completed in {round(time.time() - start_time, 2)} seconds")

st.dataframe(semantic_df)

# Export semantic matrix
csv_data = semantic_df.to_csv(index=False).encode('utf-8')
st.download_button("Download Semantic Mapping", data=csv_data, file_name="semantic_mapping.csv")
