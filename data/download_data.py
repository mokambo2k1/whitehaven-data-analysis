"""
download_data.py
-----------------
Downloads both datasets used in this analysis from public GitHub mirrors
(no Kaggle login required). Run this once before the analysis scripts.

Usage:
    python src/download_data.py
"""

import os
import ssl
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

FILES = {
    "ai4i2020.csv": (
        "https://raw.githubusercontent.com/lingchenlijing/"
        "AI4I-2020-Predictive-Maintenance-Dataset/master/data/ai4i2020.csv"
    ),
    "industrial_safety.csv": (
        "https://raw.githubusercontent.com/aws-samples/"
        "amazon-sagemaker-sentence-similarity-hugging-face/main/"
        "IHMStefanini_industrial_safety_and_health_database_with_accidents_description.csv"
    ),
}


def download(url: str, dest: str) -> None:
    """Download url to dest, with a browser-like User-Agent (some
    networks / corporate proxies block the default urllib agent) and
    an SSL fallback in case of a local certificate-store issue, which
    is common on Windows/Anaconda installs."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req) as resp, open(dest, "wb") as out:
            out.write(resp.read())
    except urllib.error.URLError as e:
        if isinstance(e.reason, ssl.SSLError):
            # Fallback: retry without certificate verification.
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            with urllib.request.urlopen(req, context=ctx) as resp, open(dest, "wb") as out:
                out.write(resp.read())
        else:
            raise


for filename, url in FILES.items():
    dest = os.path.join(DATA_DIR, filename)
    print(f"Downloading {filename} ...")
    print(f"  URL: {url}")
    try:
        download(url, dest)
        size = os.path.getsize(dest)
        print(f"  saved to {dest} ({size:,} bytes)")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        print(f"  If this keeps failing, open the URL above directly in your "
              f"browser, save the page as '{filename}', and place it in the "
              f"data/ folder manually.")

print("\nOriginal sources (for citation in your report):")
print("  AI4I 2020: https://www.kaggle.com/datasets/geetanjalisikarwar/equipment-failure-prediction-dataset")
print("  Industrial Safety: https://www.kaggle.com/datasets/ihmstefanini/industrial-safety-and-health-analytics-database")
