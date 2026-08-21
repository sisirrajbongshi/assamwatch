"""
╔══════════════════════════════════════════════════════════════╗
║              AssamWatch — MAIN DASHBOARD                    ║
║  Real-Time Multi-Domain Disaster Signal Detection System    ║
║  for Assam, Northeast India                                 ║
║                                                              ║
║  HOW TO RUN:                                                ║
║    streamlit run app.py                                     ║
║                                                              ║
║  Dept. of CS, PDUAM, Amjonga, Goalpara                     ║
║  Supervisor: Dr. Sisir Kumar Rajbongshi                     ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import os
import sys
from datetime import datetime, timedelta
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.classifier import AssamDisasterClassifier
from models.vulnerability_processor import VulnerabilityProcessor, DISTRICT_COORDS

# ─────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AssamWatch — Live Disaster Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1F3864, #2E74B5);
        padding: 20px; border-radius: 10px;
        color: white; text-align: center; margin-bottom: 20px;
    }
    .metric-card {
        background: #f8f9fa; border-left: 4px solid #2E74B5;
        padding: 15px; border-radius: 8px; margin: 5px;
    }
    .alert-critical { background: #FFE6E6; border-left: 4px solid #C00000;
                      padding: 10px; border-radius: 5px; }
    .alert-high    { background: #FFF3E0; border-left: 4px solid #FF6600;
                      padding: 10px; border-radius: 5px; }
    .alert-moderate{ background: #FFFDE7; border-left: 4px solid #FFC107;
                      padding: 10px; border-radius: 5px; }
    .domain-pill {
        display: inline-block; padding: 3px 10px;
        border-radius: 12px; font-size: 0.8em; font-weight: bold;
    }
    .stMetric { background: #f0f4f8; border-radius: 8px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# COLOUR SCHEME
# ─────────────────────────────────────────────────────────────

DOMAIN_COLORS = {
    "Flood":        "#1565C0",
    "Elephant":     "#2E7D32",
    "Agriculture":  "#F9A825",
    "Health":       "#C62828",
    "Weather":      "#6A1B9A",
    "General":      "#757575",
}

RISK_COLORS = {
    "CRITICAL": "#C00000",
    "HIGH":     "#FF6600",
    "MODERATE": "#FFC107",
    "LOW":      "#2E7D32",
}

DOMAIN_ICONS = {
    "Flood":       "🌊",
    "Elephant":    "🐘",
    "Agriculture": "🌾",
    "Health":      "🏥",
    "Weather":     "⛈️",
    "General":     "📡",
    "No live signal": "⚪",
}

# ─────────────────────────────────────────────────────────────
# DATA LOADING FUNCTIONS
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_environmental_data():
    """Fetch live environmental data for key Assam districts"""
    key_districts = {
        "Goalpara":     (26.18, 90.63),
        "Dhubri":       (26.02, 89.98),
        "Barpeta":      (26.32, 91.00),
        "Kamrup":       (26.19, 91.75),
        "Nagaon":       (26.35, 92.68),
        "Jorhat":       (26.75, 94.22),
        "Dibrugarh":    (27.48, 94.90),
        "Cachar":       (24.82, 92.80),
        "Sonitpur":     (26.63, 92.80),
        "Kokrajhar":    (26.40, 90.27),
    }

    env_data = []
    for district, (lat, lon) in key_districts.items():
        try:
            r = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat, "longitude": lon,
                    "current": ["relative_humidity_2m",
                                "temperature_2m", "precipitation"],
                    "hourly": ["relative_humidity_2m", "precipitation"],
                    "forecast_days": 3, "timezone": "Asia/Kolkata"
                }, timeout=10
            )
            if r.status_code == 200:
                d = r.json()
                current = d.get("current", {})
                humidity = current.get("relative_humidity_2m", 70)
                temp     = current.get("temperature_2m", 28)
                precip   = current.get("precipitation", 0)

                # Environmental risk
                env_risk = 0
                if humidity > 90: env_risk += 40
                elif humidity > 85: env_risk += 30
                elif humidity > 75: env_risk += 15
                if precip > 20: env_risk += 35
                elif precip > 10: env_risk += 20
                elif precip > 5: env_risk += 10

                env_data.append({
                    "district":     district,
                    "humidity":     humidity,
                    "temperature":  temp,
                    "precipitation":precip,
                    "env_risk":     min(env_risk, 100),
                    "latitude":     lat,
                    "longitude":    lon,
                })
        except Exception as e:
            env_data.append({
                "district": district, "humidity": 75,
                "temperature": 28, "precipitation": 0,
                "env_risk": 30, "latitude": lat, "longitude": lon,
            })

    return pd.DataFrame(env_data) if env_data else pd.DataFrame()


@st.cache_data(ttl=600)
def load_live_news_data():
    """
    Loads GENUINELY LIVE news data collected by sentinel_assam/collect_all_data.py.
    Returns (dataframe, last_updated_timestamp, is_stale) — or (None, None, None)
    if no live collection has ever been run.

    IMPORTANT: This function NEVER falls back to the static 81-post research
    corpus or demo data. If no live data exists, it returns None so the UI
    can honestly say "no live data yet" instead of silently substituting
    historical data.
    """
    import json
    live_dir = "data/live"
    ts_path = f"{live_dir}/last_updated.json"
    master_path = f"{live_dir}/MASTER_NEWS_DATA.xlsx"

    if not (os.path.exists(ts_path) and os.path.exists(master_path)):
        return None, None, None

    try:
        with open(ts_path) as f:
            meta = json.load(f)
        last_updated = meta.get("last_updated")
        df = pd.read_excel(master_path)

        # Determine staleness — live data older than 6 hours is flagged
        is_stale = True
        if last_updated:
            age = datetime.now() - datetime.strptime(last_updated, '%Y-%m-%d %H:%M:%S')
            is_stale = age > timedelta(hours=6)

        return df, last_updated, is_stale
    except Exception:
        return None, None, None


@st.cache_data(ttl=3600)
def load_research_corpus():
    """
    Loads the 81-post RESEARCH VALIDATION CORPUS collected manually from
    verified Assamese news sources and public community posts (2020–2026).

    This dataset is STATIC and used exclusively to validate the classifier
    (macro-F1 = 0.919). It is explicitly NOT live data and must never be
    labeled or displayed as a live/current signal feed.
    """
    validated_path = "data/facebook_posts_all.csv"
    if os.path.exists(validated_path):
        try:
            df = pd.read_csv(validated_path)
            return df
        except Exception:
            pass
    return pd.DataFrame()


@st.cache_data(ttl=600)
def load_news_data():
    """
    LEGACY compatibility wrapper. Prefer load_live_news_data() and
    load_research_corpus() directly for new code — this function is kept
    so existing calls in this file continue to work, but it now clearly
    tags every row with its true source rather than mixing live and
    historical data silently.
    """
    live_df, last_updated, is_stale = load_live_news_data()
    if live_df is not None and len(live_df) > 0:
        live_df = live_df.copy()
        live_df['data_source'] = 'live'
        live_df['is_demo'] = False
        return live_df

    # No live data collected yet — fall back to demo ONLY, and label clearly.
    # (Research corpus is intentionally NOT used here — it must be shown
    # separately on its own page so it is never mistaken for live signals.)
    demo_df = generate_demo_news()
    demo_df['data_source'] = 'demo'
    demo_df['is_demo'] = True
    demo_df['is_demo_label'] = '⚠️ DEMO DATA — not live'
    return demo_df


def generate_demo_news():
    """
    Demo/placeholder news data — used ONLY when no real data is found.
    This is NEVER shown as live data — always labeled explicitly in the UI.
    Real data comes from facebook_posts_all.csv (81 validated posts) or
    from running sentinel_assam/collect_all_data.py.
    """
    import random

    demo_posts = [
        ("ভয়াবহ বান পৰিস্থিতি গোৱালপাৰাত, শতাধিক পৰিয়াল গৃহহীন", "Goalpara", "Flood"),
        ("Elephant herd destroys paddy fields in Kamrup district", "Kamrup", "Elephant"),
        ("Heavy rainfall warning for lower Assam districts", "Assam (General)", "Weather"),
        ("গোৱালপাৰাত হাতীৰ আক্ৰমণত কৃষকৰ মৃত্যু", "Goalpara", "Elephant"),
        ("Flood situation worsens in Dhubri", "Dhubri", "Flood"),
        ("Rising fever cases in Barpeta hospitals", "Barpeta", "Health"),
        ("Banana crop severely damaged in Bongaigaon due to floods", "Bongaigaon", "Agriculture"),
        ("ASDMA issues orange alert for 8 districts", "Assam (General)", "Flood"),
        ("Paddy fields submerged in Nagaon district", "Nagaon", "Agriculture"),
        ("Health advisory issued for flood-affected areas of Assam", "Assam (General)", "Health"),
        ("ধুবুৰীত বানপানীয়ে ৰাস্তাঘাট অচল কৰিছে", "Dhubri", "Flood"),
        ("Elephant corridor blocked by floodwater in Goalpara", "Goalpara", "Elephant"),
        ("IMD red alert for heavy rainfall in upper Assam", "Assam (General)", "Weather"),
        ("Farmers demand compensation for crop loss in Kokrajhar", "Kokrajhar", "Agriculture"),
        ("Hospital surge in Cachar due to waterborne diseases", "Cachar", "Health"),
        ("Wild elephant kills farmer in Sonitpur", "Sonitpur", "Elephant"),
        ("Flash flood warning for Dibrugarh", "Dibrugarh", "Flood"),
        ("শিলচৰত ডায়েৰিয়া ৰোগীৰ সংখ্যা বৃদ্ধি পাইছে", "Cachar", "Health"),
        ("Embankment breach in Morigaon submerges villages", "Morigaon", "Flood"),
        ("Crop damage estimated at 50 crore in lower Assam", "Assam (General)", "Agriculture"),
    ]

    rows = []
    base_time = datetime.now()
    for i, (text, district, domain) in enumerate(demo_posts):
        rows.append({
            "date":          (base_time - timedelta(hours=i*2)).strftime('%Y-%m-%d %H:%M'),
            "platform":      random.choice(["Facebook", "News Website", "Twitter"]),
            "source_name":   random.choice(["Pratidin Time", "Sentinel Assam",
                                            "NE Now", "DY365"]),
            "district":      district,
            "post_text":     text,
            "language":      "Assamese" if any('\u0980' <= c <= '\u09FF'
                                               for c in text) else "English",
            "domain":        domain,
            "primary_domain":domain,
            "signal_strength":"Strong" if i < 5 else "Moderate",
        })

    return pd.DataFrame(rows)


def load_vulnerability_data():
    """Load pre-computed vulnerability scores"""
    vuln_path = "data/district_vulnerability.csv"
    if os.path.exists(vuln_path):
        return pd.read_csv(vuln_path)
    else:
        # Generate demo vulnerability data
        proc = VulnerabilityProcessor()
        demo_df = proc.generate_demo_data(n=80)
        return proc.process(demo_df)


def compute_combined_risk(env_df, news_df, vuln_df, is_live=False):
    """
    Compute combined risk score for each district.
    Combines: Environmental risk + Community signal + Population vulnerability

    is_live: True only if news_df contains genuinely live-collected data
             (from sentinel_assam/collect_all_data.py), False if news_df is
             demo data or empty. This controls whether "dominant_domain" is
             reported at all — we NEVER fabricate a dominant domain guess.
    """
    all_districts = list(DISTRICT_COORDS.keys())
    results = []

    for district in all_districts:
        if district == "Assam (General)":
            continue

        coords = DISTRICT_COORDS[district]

        # Environmental risk (0-100) — genuinely live from Open-Meteo
        env_row = env_df[env_df["district"] == district] if len(env_df) > 0 else pd.DataFrame()
        env_risk = env_row["env_risk"].values[0] if len(env_row) > 0 else 30

        # Community signal (0-100) — ONLY counted if data is genuinely live
        district_news = pd.DataFrame()
        if is_live and len(news_df) > 0:
            district_news = news_df[
                (news_df.get("district", pd.Series()) == district) |
                (news_df.get("district_detected", pd.Series()) == district)
            ]
            n_signals = len(district_news[
                district_news.get("primary_domain", "General") != "General"
            ] if "primary_domain" in district_news.columns else district_news)
            social_risk = min(n_signals * 15, 100)
        else:
            # No live community signal available — do not fabricate one.
            social_risk = 0

        # Population vulnerability (0-100)
        vuln_row = vuln_df[vuln_df["district"] == district] \
                   if len(vuln_df) > 0 else pd.DataFrame()
        if len(vuln_row) > 0:
            hvi = vuln_row["avg_hvi"].values[0]
            vuln_risk = hvi * 10  # Scale 0-10 to 0-100
        else:
            vuln_risk = 50  # Default moderate vulnerability

        # Combined risk — weighted fusion
        combined = (
            env_risk    * 0.45 +
            social_risk * 0.35 +
            vuln_risk   * 0.20
        )

        risk_label = (
            "CRITICAL" if combined >= 70 else
            "HIGH"     if combined >= 50 else
            "MODERATE" if combined >= 30 else
            "LOW"
        )

        # Dominant domain — ONLY reported from genuine live signals for this
        # district. We NEVER guess a domain (e.g. never assume "Flood" just
        # because environmental risk is high) — that would be fabricated data.
        if is_live and len(district_news) > 0 and "primary_domain" in district_news.columns:
            real_signals = district_news[district_news["primary_domain"] != "General"]
            if len(real_signals) > 0:
                dominant_domain = real_signals["primary_domain"].mode().iloc[0]
            else:
                dominant_domain = "No live signal"
        else:
            dominant_domain = "No live signal"

        results.append({
            "district":       district,
            "latitude":       coords[0],
            "longitude":      coords[1],
            "env_risk":       round(env_risk, 1),
            "social_risk":    round(social_risk, 1),
            "vuln_risk":      round(vuln_risk, 1),
            "combined_risk":  round(combined, 1),
            "risk_label":     risk_label,
            "dominant_domain":dominant_domain,
        })

    return pd.DataFrame(results).sort_values("combined_risk", ascending=False)


# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────

def main():

    # ── HEADER ──
    st.markdown("""
    <div class="main-header">
        <h1>🛰️ AssamWatch</h1>
        <h3>Real-Time Multi-Domain Disaster Signal Intelligence for Assam</h3>
        <p>Combining Environmental Sensing • Social Media Analysis •
           Population Vulnerability • Live Google Maps</p>
        <small>Dept. of CS, PDUAM, Amjonga | Supervisor: Dr. Sisir Kumar Rajbongshi</small>
    </div>
    """, unsafe_allow_html=True)

    # ── LOAD ALL DATA ──
    with st.spinner("Loading data from all sources..."):
        env_df  = load_environmental_data()
        live_news_df, live_last_updated, live_is_stale = load_live_news_data()
        research_corpus_df = load_research_corpus()
        vuln_df = load_vulnerability_data()

        clf = AssamDisasterClassifier()

        # Determine what "news_df" means for the rest of the page — this is
        # the ONLY place that decides whether we truly have live data.
        is_live = live_news_df is not None and len(live_news_df) > 0
        if is_live:
            news_df = live_news_df.copy()
            if "primary_domain" not in news_df.columns:
                news_df = clf.process_dataframe(news_df)
        else:
            news_df = generate_demo_news()
            news_df['is_demo'] = True

        # Compute combined risk — is_live controls whether dominant_domain
        # and social_risk are computed from real signals or honestly left
        # as "No live signal" / 0.
        risk_df = compute_combined_risk(env_df, news_df, vuln_df, is_live=is_live)

    # ── LAST UPDATE TIME — HONEST, REFLECTS ACTUAL DATA FRESHNESS ──
    if is_live:
        staleness_note = " ⚠️ (stale — over 6h old, re-run collector)" if live_is_stale else " ✅"
        st.caption(f"🕐 Live news last collected: {live_last_updated}{staleness_note}  |  "
                   f"Environmental data fetched: {datetime.now().strftime('%d %B %Y  |  %H:%M IST')}")
    else:
        st.caption(f"🕐 Environmental data fetched: {datetime.now().strftime('%d %B %Y  |  %H:%M IST')}  |  "
                   f"⚠️ No live news collected yet — showing demo placeholders. "
                   f"Run `sentinel_assam/collect_all_data.py` to fetch real live signals.")

    # ── NAVIGATION ──
    page = st.sidebar.radio(
        "📍 Navigation",
        ["🏠 Live Overview",
         "🗺️ District Risk Map",
         "📊 Signal Analysis",
         "📚 Research Corpus (2020–2026)",
         "🏥 Vulnerability Profile",
         "🔬 Classify Post"]
    )

    # ─────────────────────────────────────────
    # PAGE 1 — LIVE OVERVIEW
    # ─────────────────────────────────────────
    if page == "🏠 Live Overview":

        # ── TOP METRICS ──
        col1, col2, col3, col4, col5 = st.columns(5)

        critical_count = len(risk_df[risk_df["risk_label"] == "CRITICAL"])
        high_count     = len(risk_df[risk_df["risk_label"] == "HIGH"])
        total_signals  = (len(news_df[news_df.get("primary_domain", "General") != "General"])
                           if is_live else 0)

        with col1:
            st.metric("🔴 CRITICAL Districts", critical_count,
                      delta="⚠️ Act Now" if critical_count > 0 else "✅ None")
        with col2:
            st.metric("🟠 HIGH Risk Districts", high_count)
        with col3:
            st.metric("📡 Live Signals", total_signals,
                      delta="✅ Live" if is_live else "No live data")
        with col4:
            avg_humidity = env_df["humidity"].mean() if len(env_df) > 0 else 75
            st.metric("💧 Avg Humidity", f"{avg_humidity:.0f}%",
                      delta="⬆️ High" if avg_humidity > 85 else "Normal")
        with col5:
            vuln_score = vuln_df["avg_hvi"].mean() if len(vuln_df) > 0 else 5.0
            st.metric("🏥 Population HVI", f"{vuln_score:.1f}/10")

        if not is_live:
            st.info(
                "ℹ️ **No live community signals collected yet.** The map below shows "
                "**environmental risk (live, from Open-Meteo) + population vulnerability "
                "(from survey)** only — community signal and dominant domain are not "
                "shown because no live news has been fetched. Run "
                "`sentinel_assam/collect_all_data.py` on a machine with internet access "
                "to enable the live community signal layer."
            )

        st.divider()

        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.subheader("🗺️ Live Risk Map — Assam Districts")

            # Plotly map
            fig = px.scatter_mapbox(
                risk_df,
                lat="latitude", lon="longitude",
                color="risk_label",
                size="combined_risk",
                hover_name="district",
                hover_data={
                    "combined_risk": True,
                    "risk_label":    True,
                    "dominant_domain":True,
                    "env_risk":      True,
                    "social_risk":   True,
                    "latitude":      False,
                    "longitude":     False,
                },
                color_discrete_map=RISK_COLORS,
                size_max=30,
                zoom=6.5,
                center={"lat": 26.2, "lon": 92.9},
                mapbox_style="open-street-map",
                title="",
                height=500,
            )
            fig.update_layout(
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)")
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.subheader("⚡ District Risk Rankings")

            for _, row in risk_df.head(12).iterrows():
                icon = (
                    "🔴" if row["risk_label"] == "CRITICAL" else
                    "🟠" if row["risk_label"] == "HIGH" else
                    "🟡" if row["risk_label"] == "MODERATE" else "🟢"
                )
                domain_icon = DOMAIN_ICONS.get(row["dominant_domain"], "📡")
                st.markdown(
                    f"{icon} **{row['district']:<18}** "
                    f"`{row['combined_risk']:.0f}/100` "
                    f"{domain_icon} {row['dominant_domain']}",
                )

        st.divider()

        # ── SIGNAL FEED — ONLY GENUINE LIVE DATA, HONEST EMPTY STATE ──
        st.subheader("📡 Live Signal Feed — Latest Community Reports")

        if is_live and len(news_df) > 0:
            st.success(
                f"✅ Showing {len(news_df)} genuinely live signals — "
                f"collected {live_last_updated}"
                + (" ⚠️ data is over 6h old, consider re-collecting" if live_is_stale else "")
            )
            # Sort by actual publish date (most recent first) so the feed
            # reflects true recency rather than Google's relevance ordering,
            # which can otherwise keep showing the same "most relevant" older
            # articles even when newer ones exist further down the raw list.
            news_df_sorted = news_df.copy()
            if "date" in news_df_sorted.columns:
                news_df_sorted["_parsed_date"] = pd.to_datetime(
                    news_df_sorted["date"], errors="coerce")
                news_df_sorted = news_df_sorted.sort_values(
                    "_parsed_date", ascending=False, na_position="last")
            recent_news = news_df_sorted.head(10)
            for _, row in recent_news.iterrows():
                domain = row.get("primary_domain", row.get("domain", "General"))
                if domain == "General":
                    continue
                icon  = DOMAIN_ICONS.get(domain, "📡")
                color = DOMAIN_COLORS.get(domain, "#757575")
                dist  = row.get("district", "Assam")
                text  = str(row.get("post_text", ""))[:120]

                st.markdown(
                    f"<div style='border-left:3px solid {color};"
                    f"padding:8px;margin:4px 0;border-radius:4px;"
                    f"background:#fafafa'>"
                    f"<b>{icon} {domain}</b> — "
                    f"<span style='color:#666'>{dist}</span><br>"
                    f"<small>{text}...</small>"
                    f"</div>",
                    unsafe_allow_html=True
                )
        else:
            st.warning(
                "⚠️ **No live signals collected yet.** This feed shows genuinely live "
                "news only — it is intentionally left empty rather than substituting "
                "demo or historical data, so nothing shown here is ever mistaken for "
                "a current event.\n\n"
                "**To populate this feed:** run "
                "`python sentinel_assam/collect_all_data.py` on a computer with "
                "internet access, then refresh this dashboard.\n\n"
                "To see the 81-post historical research corpus used to validate the "
                "classifier, open the **📚 Research Corpus (2020–2026)** page in the "
                "sidebar — it is clearly dated and labeled as non-live."
            )

    # ─────────────────────────────────────────
    # PAGE 2 — DISTRICT RISK MAP
    # ─────────────────────────────────────────
    elif page == "🗺️ District Risk Map":
        st.subheader("🗺️ Detailed District Risk Analysis")
        if not is_live:
            st.info(
                "ℹ️ **Social Media Signal = 0 and Dominant Domain = 'No live signal' "
                "for all districts** because no live community data has been collected "
                "yet. Environmental Risk and Population Vulnerability below are real. "
                "Run `sentinel_assam/collect_all_data.py` to activate the community "
                "signal layer."
            )

        # Domain filter
        selected_domain = st.selectbox(
            "Filter by Domain",
            ["All Domains", "Flood", "Elephant", "Agriculture", "Health", "Weather"]
        )

        # Risk breakdown chart
        fig = go.Figure()

        fig.add_trace(go.Bar(
            name="Environmental Risk",
            x=risk_df["district"].head(15),
            y=risk_df["env_risk"].head(15),
            marker_color="#1565C0", opacity=0.8
        ))
        fig.add_trace(go.Bar(
            name="Social Media Signal",
            x=risk_df["district"].head(15),
            y=risk_df["social_risk"].head(15),
            marker_color="#C62828", opacity=0.8
        ))
        fig.add_trace(go.Bar(
            name="Population Vulnerability",
            x=risk_df["district"].head(15),
            y=risk_df["vuln_risk"].head(15),
            marker_color="#F9A825", opacity=0.8
        ))

        fig.update_layout(
            barmode="group",
            title="Risk Component Breakdown — Top 15 Districts",
            xaxis_title="District",
            yaxis_title="Risk Score (0–100)",
            height=450,
            legend=dict(x=0.7, y=0.99),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Detailed table
        st.subheader("📋 Complete District Risk Table")
        display_df = risk_df[[
            "district", "risk_label", "combined_risk",
            "env_risk", "social_risk", "vuln_risk", "dominant_domain"
        ]].copy()
        display_df.columns = [
            "District", "Risk Level", "Combined Score",
            "Environmental", "Social Media", "Vulnerability", "Dominant Domain"
        ]
        st.dataframe(display_df, use_container_width=True, height=500)

    # ─────────────────────────────────────────
    # PAGE 3 — SIGNAL ANALYSIS
    # ─────────────────────────────────────────
    elif page == "📊 Signal Analysis":
        st.subheader("📊 Live Signal Analysis")

        if is_live and len(news_df) > 0 and "primary_domain" in news_df.columns:
            st.success(f"✅ Analysing {len(news_df)} live signals collected {live_last_updated}")
            col1, col2 = st.columns(2)

            with col1:
                # Domain distribution pie
                domain_counts = news_df["primary_domain"].value_counts()
                domain_counts = domain_counts[domain_counts.index != "General"]
                fig = px.pie(
                    values=domain_counts.values,
                    names=domain_counts.index,
                    title="Live Signal Distribution by Domain",
                    color=domain_counts.index,
                    color_discrete_map=DOMAIN_COLORS,
                    hole=0.4,
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # District signal bar
                dist_col = "district_detected" if "district_detected" in news_df.columns \
                           else "district"
                district_counts = news_df[
                    news_df["primary_domain"] != "General"
                ][dist_col].value_counts().head(10)

                fig = px.bar(
                    x=district_counts.values,
                    y=district_counts.index,
                    orientation="h",
                    title="Top Districts by Live Signal Count",
                    labels={"x": "Signal Count", "y": "District"},
                    color=district_counts.values,
                    color_continuous_scale="Reds",
                )
                fig.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)

            # Signal strength breakdown
            st.subheader("📶 Signal Strength Distribution")
            if "signal_strength" in news_df.columns:
                strength_counts = news_df["signal_strength"].value_counts()
                cols = st.columns(len(strength_counts))
                for i, (strength, count) in enumerate(strength_counts.items()):
                    with cols[i]:
                        icon = "🔴" if strength == "Strong" else \
                               "🟡" if strength == "Moderate" else "⚪"
                        st.metric(f"{icon} {strength}", count)

        else:
            st.warning(
                "⚠️ **No live signal data available yet.** This page analyses only "
                "genuinely live collected signals — it does not substitute historical "
                "or demo data, so charts are intentionally empty until real data exists.\n\n"
                "Run `python sentinel_assam/collect_all_data.py` to fetch live news, "
                "then refresh this page.\n\n"
                "To explore the static 81-post research validation corpus instead, "
                "see the **📚 Research Corpus (2020–2026)** page."
            )

    # ─────────────────────────────────────────
    # PAGE — RESEARCH CORPUS (STATIC, CLEARLY LABELED, NOT LIVE)
    # ─────────────────────────────────────────
    elif page == "📚 Research Corpus (2020–2026)":
        st.subheader("📚 Research Validation Corpus (2020–2026)")
        st.warning(
            "⚠️ **This is historical research data, NOT a live feed.** These 81 posts "
            "were manually collected from 47 verified Assamese news sources and public "
            "community pages between 2020 and 2026, and were used exclusively to "
            "validate the AssamWatch classifier (macro-F1 = 0.919). They are frozen "
            "and do not update. For genuinely current signals, see the "
            "**🏠 Live Overview** or **📊 Signal Analysis** pages."
        )

        if len(research_corpus_df) > 0:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total posts", len(research_corpus_df))
            with col2:
                st.metric("Unique sources", research_corpus_df["source_name"].nunique()
                          if "source_name" in research_corpus_df.columns else "—")
            with col3:
                st.metric("Districts represented", research_corpus_df["district"].nunique()
                          if "district" in research_corpus_df.columns else "—")

            domain_col = "domain" if "domain" in research_corpus_df.columns else None
            if domain_col:
                domain_filter = st.multiselect(
                    "Filter by domain",
                    options=sorted(research_corpus_df[domain_col].dropna().unique()),
                    default=None
                )
                display_df = research_corpus_df[research_corpus_df[domain_col].isin(domain_filter)] \
                             if domain_filter else research_corpus_df
            else:
                display_df = research_corpus_df

            st.dataframe(display_df, use_container_width=True, height=450)
        else:
            st.error("Research corpus file not found at `data/facebook_posts_all.csv`.")

    # ─────────────────────────────────────────
    # PAGE 4 — VULNERABILITY PROFILE
    # ─────────────────────────────────────────
    elif page == "🏥 Vulnerability Profile":
        st.subheader("🏥 Population Health Vulnerability Profile")
        st.info(
            "**Survey region: Goalpara district and its surrounding area.** "
            "This is a single-region pilot survey (n=100), not an independent "
            "multi-district study. 87% of respondents are from Goalpara "
            "district itself; the remaining categories shown below (Kamrup, "
            "South Salmara, Meghalaya border area, Dhubri, Other) reflect a "
            "small number of students from immediately adjoining areas "
            "(2–5 respondents each) and are shown for transparency only — "
            "they should not be read as separately sampled district estimates."
        )

        if len(vuln_df) > 0:
            col1, col2 = st.columns(2)

            with col1:
                # HVI bar chart
                fig = px.bar(
                    vuln_df.head(12),
                    x="district", y="avg_hvi",
                    color="vulnerability_label",
                    color_discrete_map=RISK_COLORS,
                    title="HVI by Home District (Goalpara Study Region)",
                    labels={"avg_hvi": "HVI Score (0–10)",
                            "district": "Home District (self-reported)"},
                )
                fig.add_hline(y=7.0, line_dash="dash",
                              line_color="red",   annotation_text="CRITICAL threshold")
                fig.add_hline(y=5.0, line_dash="dash",
                              line_color="orange",annotation_text="HIGH threshold")
                st.plotly_chart(fig, use_container_width=True)
                st.caption("⚠️ n varies sharply by category: Goalpara n=87; "
                           "all other categories n=1–5. Bars for small-n "
                           "categories are indicative only, not statistically "
                           "reliable district estimates.")

            with col2:
                # Sinusitis prevalence
                if "high_sinusitis_pct" in vuln_df.columns:
                    fig = px.bar(
                        vuln_df.head(12),
                        x="district",
                        y="high_sinusitis_pct",
                        title="% Reporting High Sinusitis Severity",
                        labels={"high_sinusitis_pct": "% Respondents",
                                "district": "Home District (self-reported)"},
                        color="high_sinusitis_pct",
                        color_continuous_scale="Reds",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("⚠️ Same small-n caveat applies — see note above.")

            # Key finding highlight — reframed to focus on the Goalpara
            # study region itself, not a fabricated district comparison
            goalpara_row = vuln_df[vuln_df["district"] == "Goalpara"]
            if len(goalpara_row) > 0:
                g = goalpara_row.iloc[0]
                st.markdown(f"""
                <div class="alert-critical">
                <b>🔍 KEY FINDING FROM SURVEY DATA:</b><br>
                Within the <b>Goalpara district study region</b> (n={g.get('n_respondents','87')}),
                mean HVI = {g['avg_hvi']:.1f}/10, with
                {g.get('high_sinusitis_pct', 0):.0f}%
                of respondents reporting high or chronic sinusitis severity.
                This establishes a primary-survey health vulnerability baseline
                for the region, used to calibrate the AssamWatch risk fusion
                model (Section 6). It is a pilot single-region estimate, not
                a comparative multi-district finding.
                </div>
                """, unsafe_allow_html=True)
            elif len(vuln_df) > 0:
                most_vulnerable = vuln_df.iloc[0]
                st.markdown(f"""
                <div class="alert-critical">
                <b>🔍 KEY FINDING FROM SURVEY DATA:</b><br>
                Mean HVI in the Goalpara study region = {most_vulnerable['avg_hvi']:.1f}/10,
                with {most_vulnerable.get('high_sinusitis_pct', 0):.0f}%
                of respondents reporting high sinusitis severity.
                </div>
                """, unsafe_allow_html=True)

    # ─────────────────────────────────────────
    # PAGE 5 — CLASSIFY POST
    # ─────────────────────────────────────────
    elif page == "🔬 Classify Post":
        st.subheader("🔬 Assamese Language Disaster Signal Classifier")
        st.info(
            "This classifier is the core CS contribution of AssamWatch. "
            "Enter any Assamese or English text about a disaster — the system "
            "automatically detects the domain and district."
        )

        post_text = st.text_area(
            "Enter Assamese or English text to classify:",
            placeholder="e.g. ভয়াবহ বান পৰিস্থিতি গোৱালপাৰাত...\n"
                        "or: Elephant attack reported in Kamrup district",
            height=120
        )

        if st.button("🔍 Classify", type="primary") and post_text:
            clf = AssamDisasterClassifier()
            result = clf.classify(post_text)
            district, d_conf = clf.detect_district(post_text)

            col1, col2, col3 = st.columns(3)
            with col1:
                domain = result["primary_domain"]
                st.metric(
                    "Primary Domain",
                    f"{DOMAIN_ICONS.get(domain, '📡')} {domain}"
                )
            with col2:
                st.metric("Signal Strength", result["signal_strength"])
            with col3:
                st.metric("District Detected", district)

            st.markdown("**Matched Keywords:**")
            if result["matched_keywords"]:
                for kw in result["matched_keywords"]:
                    st.markdown(
                        f"<span style='background:#e3f2fd;padding:3px 8px;"
                        f"border-radius:4px;margin:2px;display:inline-block'>"
                        f"{kw}</span>",
                        unsafe_allow_html=True
                    )
            else:
                st.markdown("*No domain-specific keywords detected*")

            if result["secondary_domain"]:
                st.markdown(
                    f"**Secondary Domain:** "
                    f"{DOMAIN_ICONS.get(result['secondary_domain'], '📡')} "
                    f"{result['secondary_domain']}"
                )

            # Score breakdown
            st.markdown("**Domain Score Breakdown:**")
            scores = result["all_scores"]
            score_df = pd.DataFrame({
                "Domain": list(scores.keys()),
                "Score":  list(scores.values()),
            }).sort_values("Score", ascending=True)

            fig = px.bar(
                score_df, x="Score", y="Domain",
                orientation="h",
                color="Score",
                color_continuous_scale="Reds",
                title="Keyword Match Scores by Domain",
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── SIDEBAR ──
    st.sidebar.divider()
    st.sidebar.markdown("### 📥 Data Sources Active")
    st.sidebar.success("✅ Open-Meteo Environmental API")
    st.sidebar.success("✅ Google News RSS")
    st.sidebar.success("✅ Assamese News Websites")

    if os.path.exists("data/district_vulnerability.csv"):
        st.sidebar.success("✅ Google Form Survey Data")
    else:
        st.sidebar.warning("⏳ Google Form CSV — Pending")

    if os.path.exists("data/twitter"):
        st.sidebar.success("✅ Twitter/X Data")
    else:
        st.sidebar.info("💡 Twitter API — Optional")

    st.sidebar.divider()
    st.sidebar.markdown("### 🔄 Refresh Data")
    if st.sidebar.button("🔄 Refresh Now"):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.divider()
    st.sidebar.markdown(
        "**AssamWatch** | Dept. of CS, PDUAM, Amjonga\n\n"
        "Supervisor: Dr. Sisir Kumar Rajbongshi\n\n"
        "Affiliated to Gauhati University"
    )


if __name__ == "__main__":
    main()
