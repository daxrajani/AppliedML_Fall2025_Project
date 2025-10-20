# scripts/fetch_dataset.py
import pandas as pd
import requests
from pathlib import Path

out = Path("data")
out.mkdir(parents=True, exist_ok=True)

url = "https://people.dbmi.columbia.edu/~friedma/Projects/DiseaseSymptomKB/index.html"

# 1) try pandas.read_html (extracts tables from the page)
try:
    tables = pd.read_html(url)
    # pick largest table
    df = max(tables, key=lambda t: t.shape[0])
    df.to_csv(out / "columbia_disease_symptom_raw.csv", index=False)
    print("Saved table from read_html.")
except Exception as e:
    print("read_html failed:", e)
    # 2) fallback: try requests + save HTML
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        (out / "columbia_page.html").write_text(r.text, encoding="utf-8")
        print("Saved page HTML to data/columbia_page.html -- check manually.")
    except Exception as e2:
        print("fallback fetch failed:", e2)
        print("If direct fetch fails, download the dataset manually from the web page and place it into data/")
