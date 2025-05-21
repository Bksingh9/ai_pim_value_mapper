# AI Attribute + Value Mapper

This Streamlit app allows you to upload Excel files with multiple marketplace sheets (e.g., Amazon, Flipkart),
map their attribute and value pairs to a unified global taxonomy using AI, and export a final cleaned output.

## Features

- 🔁 AI attribute mapping using `all-mpnet-base-v2`
- 🧠 AI-based value mapping using marketplace with the most variety as a base
- 📦 Global attribute/value memory and persistent editing
- 📤 PIM-style export in your required format
- 🗃 Multi-sheet support for various marketplaces

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

To deploy this on [Streamlit Community Cloud](https://streamlit.io/cloud):

1. Push this repo to GitHub.
2. Login to Streamlit Cloud and select this repo.
3. Choose `app.py` as the entry point and click Deploy.
