"""
╔══════════════════════════════════════════════════════════════╗
║              AssamWatch — HOW TO RUN                        ║
╚══════════════════════════════════════════════════════════════╝

STEP 1 — Install requirements:
  pip install streamlit plotly pandas requests feedparser
             beautifulsoup4 openpyxl scikit-learn

STEP 2 — Test the classifier:
  cd assamwatch
  python models/classifier.py

STEP 3 — Process your Google Form CSV:
  Copy your CSV to: data/survey_responses.csv
  python models/vulnerability_processor.py

STEP 4 — Collect news data:
  python ../sentinel_assam/collect_all_data.py

STEP 5 — Run the dashboard:
  streamlit run app.py

STEP 6 — Deploy online (FREE):
  1. Upload to GitHub repository
  2. Go to share.streamlit.io
  3. Connect GitHub repo
  4. Click Deploy
  → Your app goes live at: https://your-app-name.streamlit.app
"""

# requirements.txt content
REQUIREMENTS = """
streamlit>=1.28.0
plotly>=5.15.0
pandas>=2.0.0
requests>=2.31.0
feedparser>=6.0.0
beautifulsoup4>=4.12.0
openpyxl>=3.1.0
scikit-learn>=1.3.0
numpy>=1.24.0
"""

with open("requirements.txt", "w") as f:
    f.write(REQUIREMENTS.strip())

print("requirements.txt created")
print("\nTo run AssamWatch:")
print("  streamlit run app.py")
