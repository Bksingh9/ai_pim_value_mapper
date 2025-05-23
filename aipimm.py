import streamlit as st
import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="AI Attribute Mapper", layout="wide")
st.title("AI Attribute Mapper2.1")

MEMORY_FILE = 'mappings_memory.csv'
GLOBAL_ATTR_FILE = 'global_attributes.csv'
VALUE_MEMORY_FILE = 'value_mappings_memory.csv'
VALUE_MAP_FILE = 'value_mapped_results.csv'
CONFIDENCE_THRESHOLD = 0.75

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

global_value_map = {}
all_values = []
raw_attrs = []

# Parse all selected sheets and build initial value_df
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
            global_value_map.setdefault(global_attr, set()).add(val)
            all_values.append({
                "Marketplace": sheet_name,
                "Marketplace Attribute": attr,
                "Global Attribute": global_attr,
                "Marketplace Value": val
            })

value_df = pd.DataFrame(all_values)

st.subheader("🔍 Parsed Marketplace Attribute-Value Pairs")
st.dataframe(value_df)

# Attribute mapping (per marketplace)
selected_market = selected_markets[0]
if st.session_state.get('last_market') != selected_market:
    st.session_state.pop('result_df', None)
st.session_state['last_market'] = selected_market

mdf = sheets[selected_market]
attr_col = mdf.columns[0]
val_col = mdf.columns[1]
mdf = mdf[mdf[attr_col].notna()]
market_attrs = mdf[attr_col].astype(str).str.strip().tolist()  # preserve duplicates

memory_df = load_memory(MEMORY_FILE, ['Marketplace', 'Marketplace Attribute', 'Global Attribute'])
market_memory = memory_df[memory_df['Marketplace'] == selected_market]
memory_lookup = dict(zip(market_memory['Marketplace Attribute'], market_memory['Global Attribute']))

st.write("📁 Loaded Marketplace Attribute Memory", market_memory)

if st.button("Generate Attribute Suggestions") or 'result_df' in st.session_state:
    if 'result_df' not in st.session_state:
        global_attrs = list(global_value_map.keys())
        g_emb = embed_texts(global_attrs) if global_attrs else []
        m_emb = embed_texts(market_attrs)

        results = []
        for idx, attr in enumerate(market_attrs):
            if attr in memory_lookup and pd.notna(memory_lookup[attr]) and memory_lookup[attr].strip():
                ai_attr = memory_lookup[attr].strip()
                mapped_attr = ai_attr
                confidence = 1.0
            elif len(g_emb):
                sims = util.cos_sim(m_emb[idx], g_emb).flatten().numpy()
                best_idx = np.argmax(sims)
                best_score = sims[best_idx]
                ai_attr = global_attrs[best_idx] if best_score >= CONFIDENCE_THRESHOLD else generate_generic_label(attr)
                mapped_attr = ""
                confidence = float(best_score)
            else:
                ai_attr = generate_generic_label(attr)
                mapped_attr = ""
                confidence = 1.0

            results.append({
                "Marketplace Attribute": attr,
                "AI Attribute (Suggested)": ai_attr,
                "Mapped Attribute": mapped_attr,
                "Confidence": round(confidence, 3)
            })

        st.session_state['result_df'] = pd.DataFrame(results)

    result_df = st.session_state['result_df']
    st.dataframe(result_df)

    st.subheader("📝 Edit and Save Attribute Mappings")
    edited_df = st.data_editor(result_df, num_rows="dynamic")
    st.session_state['result_df'] = edited_df

    csv = edited_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Attribute Mappings CSV", data=csv, file_name="attribute_mappings.csv", mime="text/csv")

    # Save attribute memory per marketplace
    mapped = edited_df[edited_df['Mapped Attribute'].fillna('').str.strip() != ""]
    mapped['Marketplace'] = selected_market
    new_memory = mapped[['Marketplace', 'Marketplace Attribute', 'Mapped Attribute']].rename(columns={
        'Mapped Attribute': 'Global Attribute'
    })

    combined = pd.concat([memory_df, new_memory])
    combined = combined.drop_duplicates(subset=['Marketplace', 'Marketplace Attribute'], keep='last')
    save_memory(MEMORY_FILE, combined)

    # Update global attributes file
    new_globals = edited_df[
        ~edited_df['AI Attribute (Suggested)'].isin(global_value_map.keys()) &
        edited_df['AI Attribute (Suggested)'].notna()
    ]['AI Attribute (Suggested)'].dropna().unique().tolist()

    global_df = load_memory(GLOBAL_ATTR_FILE, ['Global Attribute'])
    global_df = pd.concat([global_df, pd.DataFrame({"Global Attribute": new_globals})]).drop_duplicates()
    save_memory(GLOBAL_ATTR_FILE, global_df)

# AI-based value mapping (marketplace-aware)
st.subheader("🧠 AI Value Mapping Across Marketplaces")

base_map = (
    value_df.groupby(['Global Attribute', 'Marketplace'])['Marketplace Value']
    .nunique().reset_index()
)
base_map = base_map.sort_values(['Global Attribute', 'Marketplace Value'], ascending=[True, False])
base_market_lookup = base_map.drop_duplicates('Global Attribute').set_index('Global Attribute')['Marketplace']

mapped_values = []

for attr in value_df['Global Attribute'].unique():
    base_market = base_market_lookup.get(attr)
    base_vals = value_df[
        (value_df['Global Attribute'] == attr) &
        (value_df['Marketplace'] == base_market)
    ]['Marketplace Value'].dropna().unique().tolist()

    base_embs = embed_texts(base_vals)

    for _, row in value_df[value_df['Global Attribute'] == attr].iterrows():
        mp_val = row['Marketplace Value']
        mp_emb = embed_texts([mp_val])[0]
        sims = util.cos_sim(mp_emb, base_embs).flatten().numpy()
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        global_val = base_vals[best_idx] if best_score >= CONFIDENCE_THRESHOLD else mp_val
        match_type = 'AI' if best_score >= CONFIDENCE_THRESHOLD else 'Original'

        mapped_values.append({
            "Marketplace": row['Marketplace'],
            "Marketplace Attribute": row['Marketplace Attribute'],
            "Global Attribute": attr,
            "Marketplace Value": mp_val,
            "Global Value": global_val,
            "Confidence": round(best_score, 3),
            "Match Type": match_type
        })

value_map_df = pd.DataFrame(mapped_values)
st.subheader("✅ Final Value Mapping with Global Values")
st.dataframe(value_map_df)

csv_valmap = value_map_df.to_csv(index=False).encode('utf-8')
st.download_button("Download Final Value Mapping CSV", data=csv_valmap, file_name="final_value_mapping.csv", mime="text/csv")

# Save value mapping memory
save_memory(VALUE_MEMORY_FILE, value_df.drop_duplicates())
save_memory("value_mapped_results.csv", value_map_df)
