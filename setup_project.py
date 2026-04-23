"""
Run this from your MiniProject root:
    python setup_project.py

It will create/overwrite:
  - utils/contact_checker.py
  - utils/keyword_analyzer.py
  - app.py
  - templates/index.html
  - templates/result.html
  - static/style.css
"""

import os

BASE = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
CONTACT_CHECKER = r'''import re

SUSPICIOUS_EMAIL_DOMAINS = [
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
    "rediffmail.com", "ymail.com", "mail.com"
]

def check_contact_info(text: str) -> dict:
    text_lower = text.lower()
    flags  = []
    score  = 0.0

    emails = re.findall(r"[\w.+-]+@[\w-]+\.[\w.]+", text_lower)
    for email in emails:
        domain = email.split("@")[-1]
        if domain in SUSPICIOUS_EMAIL_DOMAINS:
            flags.append(f"Personal email used for hiring: {email}")
            score += 0.3

    if "whatsapp" in text_lower:
        flags.append("WhatsApp contact — unusual for legitimate companies")
        score += 0.25

    phones = re.findall(r"\b[+]?[0-9]{10,13}\b", text)
    if phones:
        flags.append(f"Direct phone number in posting: {phones[0]}")
        score += 0.15

    if "telegram" in text_lower:
        flags.append("Telegram contact — unusual for legitimate companies")
        score += 0.2

    return {
        "flags":        flags,
        "risk_score":   min(score, 1.0),
        "emails_found": emails,
        "phones_found": phones
    }
'''

# ─────────────────────────────────────────────────────────────────────────────
KEYWORD_ANALYZER = r'''import re
from collections import Counter

MONEY_KEYWORDS   = ["earn","salary","income","payment","paid","weekly","daily","monthly","guaranteed","unlimited"]
URGENCY_KEYWORDS = ["urgent","immediately","asap","limited seats","apply now","hurry","last date","deadline","today only"]
VAGUE_KEYWORDS   = ["good communication","team player","dynamic","passionate","self motivated","go-getter","multitask"]

def analyze_keywords(text: str) -> dict:
    text_lower = text.lower()
    words      = re.findall(r"\b\w+\b", text_lower)
    total      = len(words)

    money_hits   = [w for w in MONEY_KEYWORDS   if w in text_lower]
    urgency_hits = [w for w in URGENCY_KEYWORDS if w in text_lower]
    vague_hits   = [w for w in VAGUE_KEYWORDS   if w in text_lower]

    counter            = Counter(words)
    top_word, top_count = counter.most_common(1)[0] if counter else ("", 0)
    density            = round(top_count / total * 100, 1) if total > 0 else 0
    is_stuffed         = density > 5.0

    risk  = 0.0
    risk += len(money_hits)   * 0.08
    risk += len(urgency_hits) * 0.10
    risk += len(vague_hits)   * 0.04
    if is_stuffed:
        risk += 0.2

    return {
        "money_keywords":   money_hits,
        "urgency_keywords": urgency_hits,
        "vague_keywords":   vague_hits,
        "top_word":         top_word,
        "top_density":      density,
        "is_stuffed":       is_stuffed,
        "risk_score":       min(risk, 1.0),
        "total_words":      total
    }
'''

# ─────────────────────────────────────────────────────────────────────────────
APP_PY = r'''import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from flask import Flask, render_template, request
from model.predict import predict
from explainability.shap_explainer import get_top_shap_words
from utils.scraper import scrape_job_url
from utils.pdf_parser import extract_text_from_pdf, save_uploaded_pdf
from utils.contact_checker import check_contact_info
from utils.keyword_analyzer import analyze_keywords

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)
app.config["UPLOAD_FOLDER"]      = os.path.join(BASE_DIR, "uploads")
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() == "pdf"

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    text       = ""
    source     = "Text input"
    input_type = request.form.get("input_type", "text")

    if input_type == "url":
        url    = request.form.get("url", "").strip()
        result = scrape_job_url(url)
        if not result["success"]:
            return render_template("index.html", error=result["error"])
        text   = result["text"]
        source = f"Scraped from {result['source']}"

    elif input_type == "pdf":
        file = request.files.get("pdf_file")
        if not file or file.filename == "":
            return render_template("index.html", error="No PDF file selected.")
        if not allowed_file(file.filename):
            return render_template("index.html", error="Only PDF files are accepted.")
        save_path  = save_uploaded_pdf(file, app.config["UPLOAD_FOLDER"])
        pdf_result = extract_text_from_pdf(save_path)
        if not pdf_result["success"]:
            return render_template("index.html", error=pdf_result["error"])
        text   = pdf_result["text"]
        source = f"PDF upload ({pdf_result['pages']} page(s))"

    else:
        text = request.form.get("job_text", "").strip()
        if len(text) < 50:
            return render_template("index.html", error="Please enter at least 50 characters.")

    try:
        prediction = predict(text)
    except Exception as e:
        return render_template("index.html", error=f"Prediction error: {str(e)}")

    try:
        shap_result = get_top_shap_words(text)
    except Exception:
        shap_result = {"html": text, "fake_words": [], "real_words": []}

    contact  = check_contact_info(text)
    keywords = analyze_keywords(text)

    return render_template(
        "result.html",
        source     = source,
        prediction = prediction,
        shap_html  = shap_result["html"],
        fake_words = shap_result["fake_words"][:8],
        real_words = shap_result["real_words"][:8],
        contact    = contact,
        keywords   = keywords,
    )

if __name__ == "__main__":
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    app.run(debug=True)
'''

# ─────────────────────────────────────────────────────────────────────────────
INDEX_HTML = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Fake Job Detector</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#0f0f0f;color:#e8e8e8;min-height:100vh;display:flex;align-items:center;justify-content:center;padding:20px}
.wrapper{width:100%;max-width:680px}
.hero{text-align:center;margin-bottom:36px}
.hero h1{font-size:32px;font-weight:700;letter-spacing:-0.5px;color:#fff;margin-bottom:10px}
.hero p{color:#888;font-size:15px}
.badge{display:inline-block;background:#1a1a1a;border:1px solid #2a2a2a;border-radius:20px;padding:4px 14px;font-size:12px;color:#666;margin-bottom:20px;letter-spacing:0.5px}
.card{background:#161616;border:1px solid #222;border-radius:16px;padding:28px}
.tabs{display:flex;gap:6px;margin-bottom:24px;background:#0f0f0f;border-radius:10px;padding:4px}
.tab{flex:1;padding:9px;border:none;border-radius:8px;font-size:13px;font-weight:500;cursor:pointer;background:transparent;color:#666;transition:all 0.2s}
.tab.active{background:#1f1f1f;color:#fff;box-shadow:0 1px 3px rgba(0,0,0,0.4)}
.tab-content{display:none}
.tab-content.active{display:block}
textarea,input[type=url]{width:100%;background:#0f0f0f;border:1px solid #2a2a2a;border-radius:10px;padding:14px;font-size:14px;font-family:inherit;color:#e8e8e8;resize:vertical;outline:none;transition:border-color 0.2s}
textarea:focus,input[type=url]:focus{border-color:#444}
textarea{min-height:200px}
.hint{font-size:12px;color:#555;margin-top:8px}
.upload-zone{border:2px dashed #2a2a2a;border-radius:12px;padding:48px 20px;text-align:center;cursor:pointer;transition:border-color 0.2s;background:#0f0f0f}
.upload-zone:hover{border-color:#444}
.upload-icon{font-size:32px;margin-bottom:12px;color:#444}
.upload-zone p{color:#666;font-size:14px}
.btn{width:100%;margin-top:20px;padding:14px;background:#fff;color:#000;border:none;border-radius:10px;font-size:15px;font-weight:600;cursor:pointer;transition:opacity 0.2s;letter-spacing:0.2px}
.btn:hover{opacity:0.9}
.error{background:#1a0a0a;border:1px solid #3a1a1a;color:#f87171;padding:12px 16px;border-radius:10px;font-size:13px;margin-bottom:16px}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:28px}
.stat{background:#161616;border:1px solid #222;border-radius:10px;padding:14px;text-align:center}
.stat-num{font-size:22px;font-weight:700;color:#fff}
.stat-label{font-size:11px;color:#555;margin-top:4px;text-transform:uppercase;letter-spacing:0.5px}
</style>
</head>
<body>
<div class="wrapper">
  <div class="hero">
    <div class="badge">AI-POWERED ANALYSIS</div>
    <h1>Fake Job Detector</h1>
    <p>Paste a description, drop a URL, or upload a PDF to instantly check if a job posting is legitimate.</p>
  </div>

  {% if error %}
  <div class="error">{{ error }}</div>
  {% endif %}

  <div class="card">
    <div class="tabs">
      <button class="tab active" onclick="switchTab('text',this)">Paste Text</button>
      <button class="tab" onclick="switchTab('url',this)">Job URL</button>
      <button class="tab" onclick="switchTab('pdf',this)">Upload PDF</button>
    </div>

    <form action="/analyze" method="POST" enctype="multipart/form-data">
      <input type="hidden" name="input_type" id="itype" value="text"/>

      <div id="tab-text" class="tab-content active">
        <textarea name="job_text" placeholder="Paste the full job description here..."></textarea>
      </div>

      <div id="tab-url" class="tab-content">
        <input type="url" name="url" placeholder="https://www.indeed.com/viewjob?jk=..."/>
        <p class="hint">Works with Indeed, Naukri, SimplyHired and most job boards. LinkedIn is not supported.</p>
      </div>

      <div id="tab-pdf" class="tab-content">
        <div class="upload-zone" onclick="document.getElementById('pdf_file').click()">
          <div class="upload-icon">&#8679;</div>
          <p>Click to select a PDF file</p>
          <p class="hint" style="margin-top:6px">Max 5MB &middot; Job description PDF only</p>
          <input type="file" id="pdf_file" name="pdf_file" accept=".pdf" style="display:none"
            onchange="document.getElementById('fname').textContent='Selected: '+this.files[0].name"/>
        </div>
        <p id="fname" class="hint" style="margin-top:8px;text-align:center"></p>
      </div>

      <button type="submit" class="btn">Analyze Job Posting &rarr;</button>
    </form>
  </div>

  <div class="stats">
    <div class="stat"><div class="stat-num">2</div><div class="stat-label">ML Models</div></div>
    <div class="stat"><div class="stat-num">5</div><div class="stat-label">Risk Signals</div></div>
    <div class="stat"><div class="stat-num">SHAP</div><div class="stat-label">Explainable AI</div></div>
  </div>
</div>

<script>
function switchTab(name,btn){
  ['text','url','pdf'].forEach(t=>{
    document.getElementById('tab-'+t).classList.remove('active');
  });
  document.getElementById('tab-'+name).classList.add('active');
  document.getElementById('itype').value=name;
  document.querySelectorAll('.tab').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
}
</script>
</body>
</html>
'''

# ─────────────────────────────────────────────────────────────────────────────
RESULT_HTML = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Result &mdash; Fake Job Detector</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#0f0f0f;color:#e8e8e8;min-height:100vh;padding:32px 20px}
.wrapper{max-width:780px;margin:0 auto}
.topbar{display:flex;justify-content:space-between;align-items:center;margin-bottom:28px}
.topbar h1{font-size:18px;font-weight:600;color:#fff}
.back{font-size:13px;color:#555;text-decoration:none;border:1px solid #222;padding:6px 14px;border-radius:8px;transition:all 0.2s}
.back:hover{color:#fff;border-color:#444}

/* Verdict */
.verdict{border-radius:16px;padding:28px;text-align:center;margin-bottom:20px;position:relative;overflow:hidden}
.verdict.safe{background:#0a1a0a;border:1px solid #1a3a1a}
.verdict.suspicious{background:#1a1500;border:1px solid #3a2d00}
.verdict.high-risk{background:#1a0a0a;border:1px solid #3a1010}
.verdict-icon{font-size:40px;margin-bottom:10px}
.verdict-label{font-size:30px;font-weight:800;letter-spacing:-0.5px;margin-bottom:6px}
.safe .verdict-label{color:#4ade80}
.suspicious .verdict-label{color:#fbbf24}
.high-risk .verdict-label{color:#f87171}
.verdict-score{font-size:15px;color:#666;margin-bottom:4px}
.verdict-source{font-size:12px;color:#444}

/* Gauge */
.gauge-wrap{margin:20px auto 0;width:180px}
.gauge-label{font-size:12px;color:#555;text-align:center;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px}
.gauge-bar{height:8px;background:#1a1a1a;border-radius:4px;overflow:hidden}
.gauge-fill{height:100%;border-radius:4px;transition:width 1s ease}
.safe .gauge-fill{background:linear-gradient(90deg,#166534,#4ade80)}
.suspicious .gauge-fill{background:linear-gradient(90deg,#92400e,#fbbf24)}
.high-risk .gauge-fill{background:linear-gradient(90deg,#7f1d1d,#f87171)}

/* Cards */
.card{background:#161616;border:1px solid #222;border-radius:14px;padding:22px;margin-bottom:16px}
.card-title{font-size:13px;font-weight:600;color:#888;text-transform:uppercase;letter-spacing:0.6px;margin-bottom:18px}

/* Score breakdown */
.score-row{display:flex;align-items:center;gap:12px;margin-bottom:12px}
.score-name{width:170px;font-size:13px;color:#aaa;flex-shrink:0}
.score-track{flex:1;background:#0f0f0f;border-radius:4px;height:6px}
.score-bar{height:6px;border-radius:4px;background:#e24b4a;transition:width 0.8s ease}
.score-pct{width:44px;text-align:right;font-size:12px;color:#666}

/* Chips */
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}
.chip{font-size:12px;padding:4px 12px;border-radius:20px;font-weight:500}
.chip-red{background:#1a0808;border:1px solid #3a1515;color:#f87171}
.chip-amber{background:#1a1200;border:1px solid #3a2800;color:#fbbf24}
.chip-gray{background:#1a1a1a;border:1px solid #2a2a2a;color:#888}

/* SHAP text */
.shap-box{background:#0f0f0f;border-radius:10px;padding:16px;font-size:14px;line-height:2.2;margin-bottom:18px;border:1px solid #1a1a1a}
.legend{display:flex;gap:20px;font-size:12px;color:#555;margin-bottom:14px}
.leg-dot{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:middle}

/* Word bars */
.word-grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}
.word-section-title{font-size:12px;font-weight:600;margin-bottom:10px;text-transform:uppercase;letter-spacing:0.5px}
.word-section-title.red{color:#f87171}
.word-section-title.green{color:#4ade80}
.word-row{display:flex;align-items:center;gap:8px;margin-bottom:8px}
.word-name{width:100px;font-size:12px;font-family:monospace;color:#aaa;flex-shrink:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.word-track{flex:1;background:#0f0f0f;border-radius:3px;height:5px}
.word-bar-r{height:5px;border-radius:3px;background:#f87171}
.word-bar-g{height:5px;border-radius:3px;background:#4ade80}
.word-val{font-size:11px;color:#444;width:44px;text-align:right}

/* Confidence grid */
.conf-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
.conf-box{background:#0f0f0f;border:1px solid #1a1a1a;border-radius:10px;padding:14px;text-align:center}
.conf-num{font-size:22px;font-weight:700;color:#fff;margin-bottom:4px}
.conf-lbl{font-size:11px;color:#555;text-transform:uppercase;letter-spacing:0.4px}

/* Contact & keyword flags */
.flag-row{display:flex;align-items:flex-start;gap:10px;padding:10px 0;border-bottom:1px solid #1a1a1a;font-size:13px;color:#aaa}
.flag-row:last-child{border-bottom:none}
.flag-dot{width:6px;height:6px;border-radius:50%;background:#f87171;margin-top:5px;flex-shrink:0}

@media(max-width:600px){
  .word-grid,.conf-grid{grid-template-columns:1fr}
  .score-name{width:130px}
}
</style>
</head>
<body>
<div class="wrapper">

  <div class="topbar">
    <h1>Fake Job Detector</h1>
    <a href="/" class="back">&larr; Analyze another</a>
  </div>

  <!-- Verdict -->
  <div class="verdict {{ prediction.risk_label | lower | replace(' ','-') }}">
    {% if prediction.risk_label == 'Safe' %}
      <div class="verdict-icon">&#10003;</div>
    {% elif prediction.risk_label == 'Suspicious' %}
      <div class="verdict-icon">&#9888;</div>
    {% else %}
      <div class="verdict-icon">&#10007;</div>
    {% endif %}
    <div class="verdict-label">{{ prediction.risk_label }}</div>
    <div class="verdict-score">Overall risk score: {{ (prediction.final_score * 100) | round(1) }}%</div>
    <div class="verdict-source">Source: {{ source }}</div>
    <div class="gauge-wrap">
      <div class="gauge-label">Risk level</div>
      <div class="gauge-bar">
        <div class="gauge-fill" style="width:{{ (prediction.final_score * 100)|round }}%"></div>
      </div>
    </div>
  </div>

  <!-- Score breakdown -->
  <div class="card">
    <div class="card-title">Score breakdown</div>

    <div class="score-row">
      <div class="score-name">ML model (LR + NB)</div>
      <div class="score-track"><div class="score-bar" style="width:{{ (prediction.ml_score*100)|round }}%"></div></div>
      <div class="score-pct">{{ (prediction.ml_score*100)|round(1) }}%</div>
    </div>
    <div class="score-row">
      <div class="score-name">Suspicious phrases</div>
      <div class="score-track"><div class="score-bar" style="width:{{ (prediction.phrase_score*100)|round }}%"></div></div>
      <div class="score-pct">{{ (prediction.phrase_score*100)|round(1) }}%</div>
    </div>
    <div class="score-row">
      <div class="score-name">Salary anomaly</div>
      <div class="score-track"><div class="score-bar" style="width:{{ (prediction.salary.risk_score*100)|round }}%"></div></div>
      <div class="score-pct">{{ (prediction.salary.risk_score*100)|round(1) }}%</div>
    </div>
    <div class="score-row">
      <div class="score-name">Contact red flags</div>
      <div class="score-track"><div class="score-bar" style="width:{{ (contact.risk_score*100)|round }}%"></div></div>
      <div class="score-pct">{{ (contact.risk_score*100)|round(1) }}%</div>
    </div>
    <div class="score-row">
      <div class="score-name">Keyword analysis</div>
      <div class="score-track"><div class="score-bar" style="width:{{ (keywords.risk_score*100)|round }}%"></div></div>
      <div class="score-pct">{{ (keywords.risk_score*100)|round(1) }}%</div>
    </div>

    {% if prediction.salary.reason %}
    <div style="margin-top:14px;background:#0f0f0f;border-radius:8px;padding:10px 14px;font-size:13px;color:#666;border:1px solid #1a1a1a">
      {{ prediction.salary.reason }}
    </div>
    {% endif %}

    {% if prediction.suspicious_phrases %}
    <div style="margin-top:14px">
      <div style="font-size:12px;color:#f87171;font-weight:600;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.5px">Suspicious phrases found</div>
      <div class="chips">
        {% for phrase in prediction.suspicious_phrases %}
        <span class="chip chip-red">{{ phrase }}</span>
        {% endfor %}
      </div>
    </div>
    {% endif %}
  </div>

  <!-- Contact red flags -->
  {% if contact.flags %}
  <div class="card">
    <div class="card-title">Contact red flags</div>
    {% for flag in contact.flags %}
    <div class="flag-row"><div class="flag-dot"></div><div>{{ flag }}</div></div>
    {% endfor %}
  </div>
  {% endif %}

  <!-- Keyword analysis -->
  <div class="card">
    <div class="card-title">Keyword analysis</div>
    <div style="font-size:13px;color:#666;margin-bottom:14px">
      Total words: <span style="color:#aaa">{{ keywords.total_words }}</span>
      &nbsp;&middot;&nbsp;
      Top word: <span style="color:#aaa;font-family:monospace">"{{ keywords.top_word }}"</span>
      at <span style="color:{% if keywords.is_stuffed %}#f87171{% else %}#4ade80{% endif %}">{{ keywords.top_density }}%</span>
      {% if keywords.is_stuffed %}<span class="chip chip-red" style="margin-left:8px">Keyword stuffing detected</span>{% endif %}
    </div>
    {% if keywords.urgency_keywords %}
    <div style="margin-bottom:10px">
      <div style="font-size:11px;color:#888;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.4px">Urgency words</div>
      <div class="chips">{% for w in keywords.urgency_keywords %}<span class="chip chip-red">{{ w }}</span>{% endfor %}</div>
    </div>
    {% endif %}
    {% if keywords.money_keywords %}
    <div style="margin-bottom:10px">
      <div style="font-size:11px;color:#888;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.4px">Money words</div>
      <div class="chips">{% for w in keywords.money_keywords %}<span class="chip chip-amber">{{ w }}</span>{% endfor %}</div>
    </div>
    {% endif %}
    {% if keywords.vague_keywords %}
    <div>
      <div style="font-size:11px;color:#888;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.4px">Vague words</div>
      <div class="chips">{% for w in keywords.vague_keywords %}<span class="chip chip-gray">{{ w }}</span>{% endfor %}</div>
    </div>
    {% endif %}
  </div>

  <!-- SHAP highlights -->
  <div class="card">
    <div class="card-title">Word-level analysis (SHAP)</div>
    <div class="legend">
      <span><span class="leg-dot" style="background:rgba(248,113,113,0.7)"></span>Pushes toward fake</span>
      <span><span class="leg-dot" style="background:rgba(74,222,128,0.7)"></span>Pushes toward real</span>
      <span style="font-size:11px">Darker = stronger signal. Hover to see score.</span>
    </div>
    <div class="shap-box">{{ shap_html | safe }}</div>

    <div class="word-grid">
      {% if fake_words %}
      <div>
        <div class="word-section-title red">Top fake signals</div>
        {% for word, val in fake_words %}
        <div class="word-row">
          <div class="word-name" title="{{ word }}">{{ word }}</div>
          <div class="word-track"><div class="word-bar-r" style="width:{{ (val*100)|round }}%"></div></div>
          <div class="word-val">{{ "%.3f"|format(val) }}</div>
        </div>
        {% endfor %}
      </div>
      {% endif %}
      {% if real_words %}
      <div>
        <div class="word-section-title green">Top real signals</div>
        {% for word, val in real_words %}
        <div class="word-row">
          <div class="word-name" title="{{ word }}">{{ word }}</div>
          <div class="word-track"><div class="word-bar-g" style="width:{{ (val*100*-1)|round }}%"></div></div>
          <div class="word-val">{{ "%.3f"|format(val) }}</div>
        </div>
        {% endfor %}
      </div>
      {% endif %}
    </div>
  </div>

  <!-- Model confidence -->
  <div class="card">
    <div class="card-title">Model confidence</div>
    <div class="conf-grid">
      <div class="conf-box">
        <div class="conf-num">{{ (prediction.lr_prob*100)|round(1) }}%</div>
        <div class="conf-lbl">Logistic Regression</div>
      </div>
      <div class="conf-box">
        <div class="conf-num">{{ (prediction.nb_prob*100)|round(1) }}%</div>
        <div class="conf-lbl">Naive Bayes</div>
      </div>
      <div class="conf-box">
        <div class="conf-num">{{ (prediction.ml_score*100)|round(1) }}%</div>
        <div class="conf-lbl">Ensemble score</div>
      </div>
    </div>
  </div>

</div>
</body>
</html>
'''

# ─────────────────────────────────────────────────────────────────────────────
# Write all files
# ─────────────────────────────────────────────────────────────────────────────
to_write = {
    "utils/contact_checker.py":  CONTACT_CHECKER,
    "utils/keyword_analyzer.py": KEYWORD_ANALYZER,
    "app.py":                    APP_PY,
    "templates/index.html":      INDEX_HTML,
    "templates/result.html":     RESULT_HTML,
}

for rel, content in to_write.items():
    full = os.path.join(BASE, rel)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(content)
    size = os.path.getsize(full)
    print(f"  OK  {rel:40s} {size:6} bytes")

# Remove old style.css since styles are now inline
old_css = os.path.join(BASE, "static", "style.css")
if os.path.exists(old_css):
    with open(old_css, "w", encoding="utf-8") as f:
        f.write("/* styles moved inline */\n")
    print(f"  OK  static/style.css (cleared)")

print("\nAll files written. Now run:  python app.py")