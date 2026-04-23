"""
Run this from your MiniProject root:
    python setup_project_v2.py

Writes:
  - utils/job_validator.py       (new - rejects non-job text)
  - app.py                       (updated)
  - templates/index.html         (new UI with tips + loading)
  - templates/result.html        (new UI with charts)
  - templates/loading.html       (new loading page)
"""

import os

BASE = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
JOB_VALIDATOR = r'''import re

# Keywords that strongly indicate a job posting
JOB_KEYWORDS = [
    "job", "position", "role", "hiring", "vacancy", "opening",
    "apply", "application", "candidate", "applicant", "recruit",
    "resume", "cv", "interview", "salary", "compensation", "benefits",
    "experience", "qualification", "requirement", "skill", "degree",
    "responsibilities", "duties", "company", "employer", "organization",
    "full time", "part time", "remote", "work from", "office",
    "internship", "fresher", "engineer", "developer", "manager",
    "analyst", "executive", "officer", "assistant", "associate",
    "employment", "join", "team", "department", "working hours"
]

def is_job_posting(text: str) -> dict:
    """
    Checks if submitted text looks like a job posting.
    Returns:
        is_valid  (bool)
        score     (float) - how confident we are it is a job posting
        reason    (str)   - explanation if rejected
    """
    if not text or len(text.strip()) < 50:
        return {
            "is_valid": False,
            "score":    0.0,
            "reason":   "Text is too short. Please paste a complete job description (at least 50 characters)."
        }

    text_lower = text.lower()
    words      = re.findall(r"\b\w+\b", text_lower)
    total      = len(words)

    if total < 20:
        return {
            "is_valid": False,
            "score":    0.0,
            "reason":   "Too few words. Please paste the full job description, not just a title or snippet."
        }

    # Count how many job keywords appear
    hits = [kw for kw in JOB_KEYWORDS if kw in text_lower]
    score = min(len(hits) / 6.0, 1.0)   # 6+ hits = confident it's a job post

    if score < 0.25:
        return {
            "is_valid": False,
            "score":    score,
            "reason": (
                "This doesn't look like a job posting. "
                "Please paste a job description that includes details like "
                "role, responsibilities, requirements, salary, or company information."
            )
        }

    return {
        "is_valid": True,
        "score":    score,
        "reason":   ""
    }
'''

# ─────────────────────────────────────────────────────────────────────────────
APP_PY = r'''import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from flask import Flask, render_template, request, session
import json

from model.predict import predict
from explainability.shap_explainer import get_top_shap_words
from utils.scraper import scrape_job_url
from utils.pdf_parser import extract_text_from_pdf, save_uploaded_pdf
from utils.contact_checker import check_contact_info
from utils.keyword_analyzer import analyze_keywords
from utils.job_validator import is_job_posting

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)
app.secret_key = "fjd_secret_2024"
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

    # ── Get text from whichever input mode ───────────────────────────────────
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
        source = f"PDF ({pdf_result['pages']} page(s))"

    else:
        text = request.form.get("job_text", "").strip()

    # ── Validate it looks like a job posting ─────────────────────────────────
    validation = is_job_posting(text)
    if not validation["is_valid"]:
        return render_template("index.html",
            error=validation["reason"],
            input_type=input_type
        )

    # ── Store text in session then show loading page ──────────────────────────
    session["pending_text"]   = text
    session["pending_source"] = source
    return render_template("loading.html")

@app.route("/process", methods=["GET"])
def process():
    text   = session.get("pending_text", "")
    source = session.get("pending_source", "Unknown")

    if not text:
        return render_template("index.html", error="Session expired. Please try again.")

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

    # Build chart data for JS
    chart_data = {
        "labels": ["ML Model", "Phrases", "Salary", "Contact", "Keywords"],
        "values": [
            round(prediction["ml_score"] * 100, 1),
            round(prediction["phrase_score"] * 100, 1),
            round(prediction["salary"]["risk_score"] * 100, 1),
            round(contact["risk_score"] * 100, 1),
            round(keywords["risk_score"] * 100, 1),
        ]
    }

    return render_template(
        "result.html",
        source     = source,
        prediction = prediction,
        shap_html  = shap_result["html"],
        fake_words = shap_result["fake_words"][:8],
        real_words = shap_result["real_words"][:8],
        contact    = contact,
        keywords   = keywords,
        chart_data = json.dumps(chart_data),
    )

if __name__ == "__main__":
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    app.run(debug=True)
'''

# ─────────────────────────────────────────────────────────────────────────────
LOADING_HTML = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta http-equiv="refresh" content="2;url=/process"/>
<title>Analyzing...</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#0f0f0f;color:#e8e8e8;min-height:100vh;display:flex;align-items:center;justify-content:center}
.wrap{text-align:center;padding:40px}
.spinner{width:56px;height:56px;border:3px solid #222;border-top-color:#fff;border-radius:50%;animation:spin 0.8s linear infinite;margin:0 auto 28px}
@keyframes spin{to{transform:rotate(360deg)}}
h2{font-size:20px;font-weight:600;color:#fff;margin-bottom:10px}
.sub{font-size:14px;color:#555;margin-bottom:32px}
.steps{list-style:none;text-align:left;display:inline-block}
.step{display:flex;align-items:center;gap:12px;padding:8px 0;font-size:13px;color:#444;transition:color 0.4s}
.step.active{color:#e8e8e8}
.step.done{color:#4ade80}
.step-dot{width:8px;height:8px;border-radius:50%;background:#222;flex-shrink:0;transition:background 0.4s}
.step.active .step-dot{background:#fff;box-shadow:0 0 6px rgba(255,255,255,0.4)}
.step.done .step-dot{background:#4ade80}
.progress-bar{width:280px;height:3px;background:#1a1a1a;border-radius:2px;margin:28px auto 0;overflow:hidden}
.progress-fill{height:100%;width:0%;background:linear-gradient(90deg,#4ade80,#fff);border-radius:2px;transition:width 0.5s ease}
</style>
</head>
<body>
<div class="wrap">
  <div class="spinner"></div>
  <h2>Analyzing job posting...</h2>
  <p class="sub">This usually takes 2–4 seconds</p>

  <ul class="steps">
    <li class="step active" id="s1"><div class="step-dot"></div>Cleaning and preprocessing text</li>
    <li class="step"        id="s2"><div class="step-dot"></div>Running ML classification</li>
    <li class="step"        id="s3"><div class="step-dot"></div>Checking salary anomalies</li>
    <li class="step"        id="s4"><div class="step-dot"></div>Detecting contact red flags</li>
    <li class="step"        id="s5"><div class="step-dot"></div>Computing SHAP explanations</li>
  </ul>

  <div class="progress-bar">
    <div class="progress-fill" id="prog"></div>
  </div>
</div>

<script>
const steps = [1,2,3,4,5];
let current = 0;

function tick() {
  if (current > 0) {
    document.getElementById("s"+current).classList.remove("active");
    document.getElementById("s"+current).classList.add("done");
  }
  current++;
  if (current <= 5) {
    document.getElementById("s"+current).classList.add("active");
    document.getElementById("prog").style.width = (current/5*100)+"%";
    setTimeout(tick, 480);
  }
}

setTimeout(tick, 300);

// Navigate after animation completes
setTimeout(() => { window.location.href = "/process"; }, 2600);
</script>
</body>
</html>
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
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#0f0f0f;color:#e8e8e8;padding:32px 20px;min-height:100vh}
.wrapper{max-width:720px;margin:0 auto}
.hero{text-align:center;margin-bottom:36px}
.badge{display:inline-block;background:#1a1a1a;border:1px solid #2a2a2a;border-radius:20px;padding:4px 14px;font-size:11px;color:#666;margin-bottom:16px;letter-spacing:1px;text-transform:uppercase}
.hero h1{font-size:34px;font-weight:800;color:#fff;letter-spacing:-0.5px;margin-bottom:10px}
.hero p{font-size:15px;color:#666;max-width:480px;margin:0 auto;line-height:1.6}

/* Stats bar */
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:28px}
.stat{background:#161616;border:1px solid #1e1e1e;border-radius:10px;padding:14px;text-align:center}
.stat-num{font-size:20px;font-weight:700;color:#fff}
.stat-lbl{font-size:11px;color:#555;margin-top:3px;text-transform:uppercase;letter-spacing:0.4px}

/* Main card */
.card{background:#161616;border:1px solid #222;border-radius:16px;padding:26px;margin-bottom:20px}

/* Tabs */
.tabs{display:flex;gap:4px;background:#0f0f0f;border-radius:10px;padding:4px;margin-bottom:22px}
.tab{flex:1;padding:9px;border:none;border-radius:7px;font-size:13px;font-weight:500;cursor:pointer;background:transparent;color:#555;transition:all 0.2s}
.tab.active{background:#1f1f1f;color:#fff}
.tab-content{display:none}
.tab-content.active{display:block}

/* Inputs */
.textarea-wrap{position:relative}
textarea{width:100%;background:#0f0f0f;border:1px solid #222;border-radius:10px;padding:14px;font-size:14px;font-family:inherit;color:#e8e8e8;resize:vertical;outline:none;transition:border-color 0.2s;min-height:180px}
textarea:focus{border-color:#333}
.char-count{position:absolute;bottom:10px;right:14px;font-size:11px;color:#444}
input[type=url]{width:100%;background:#0f0f0f;border:1px solid #222;border-radius:10px;padding:13px 14px;font-size:14px;font-family:inherit;color:#e8e8e8;outline:none;transition:border-color 0.2s}
input[type=url]:focus{border-color:#333}
.hint{font-size:12px;color:#444;margin-top:8px;line-height:1.5}

/* Upload */
.upload-zone{border:2px dashed #222;border-radius:12px;padding:44px 20px;text-align:center;cursor:pointer;transition:all 0.2s;background:#0f0f0f}
.upload-zone:hover{border-color:#333;background:#111}
.upload-icon{font-size:28px;color:#333;margin-bottom:10px}
.upload-zone p{font-size:14px;color:#555}

/* Submit */
.btn{width:100%;margin-top:18px;padding:14px;background:#fff;color:#000;border:none;border-radius:10px;font-size:15px;font-weight:700;cursor:pointer;transition:opacity 0.2s;letter-spacing:0.2px}
.btn:hover{opacity:0.88}

/* Error */
.error{background:#160a0a;border:1px solid #2e1010;color:#f87171;padding:14px 16px;border-radius:10px;font-size:13px;margin-bottom:18px;line-height:1.5}
.error strong{display:block;margin-bottom:4px;font-size:14px}

/* Tips section */
.tips-card{background:#161616;border:1px solid #1e1e1e;border-radius:16px;padding:24px;margin-bottom:20px}
.tips-title{font-size:13px;font-weight:600;color:#666;text-transform:uppercase;letter-spacing:0.6px;margin-bottom:16px}
.tips-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.tip{background:#0f0f0f;border:1px solid #1a1a1a;border-radius:10px;padding:14px}
.tip-icon{font-size:18px;margin-bottom:7px}
.tip-title{font-size:13px;font-weight:600;color:#ccc;margin-bottom:4px}
.tip-text{font-size:12px;color:#555;line-height:1.5}

@media(max-width:600px){.stats{grid-template-columns:repeat(2,1fr)}.tips-grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="wrapper">

  <div class="hero">
    <div class="badge">AI-Powered &middot; Free &middot; Instant</div>
    <h1>Fake Job Detector</h1>
    <p>Instantly analyze any job posting for fraud signals using machine learning and explainable AI.</p>
  </div>

  <div class="stats">
    <div class="stat"><div class="stat-num">2</div><div class="stat-lbl">ML Models</div></div>
    <div class="stat"><div class="stat-num">5</div><div class="stat-lbl">Risk Signals</div></div>
    <div class="stat"><div class="stat-num">SHAP</div><div class="stat-lbl">Explainable AI</div></div>
    <div class="stat"><div class="stat-num">&lt;3s</div><div class="stat-lbl">Analysis Time</div></div>
  </div>

  {% if error %}
  <div class="error">
    <strong>Could not analyze this input</strong>
    {{ error }}
  </div>
  {% endif %}

  <div class="card">
    <div class="tabs">
      <button class="tab active" onclick="switchTab('text',this)">Paste Text</button>
      <button class="tab"        onclick="switchTab('url',this)">Job URL</button>
      <button class="tab"        onclick="switchTab('pdf',this)">Upload PDF</button>
    </div>

    <form action="/analyze" method="POST" enctype="multipart/form-data" onsubmit="return validate()">
      <input type="hidden" name="input_type" id="itype" value="text"/>

      <!-- Text -->
      <div id="tab-text" class="tab-content active">
        <div class="textarea-wrap">
          <textarea id="job_text" name="job_text"
            placeholder="Paste the full job description here — include title, responsibilities, requirements, salary and company info for best accuracy..."
            oninput="updateCount()"></textarea>
          <div class="char-count" id="charcount">0 words</div>
        </div>
        <p class="hint">Tip: paste the complete posting for best accuracy. Short snippets may not be detected correctly.</p>
      </div>

      <!-- URL -->
      <div id="tab-url" class="tab-content">
        <input type="url" name="url" id="url_input"
          placeholder="https://www.indeed.com/viewjob?jk=..."/>
        <p class="hint">Works with Indeed, Naukri, SimplyHired and most job boards. LinkedIn blocks scraping — paste the text manually instead.</p>
      </div>

      <!-- PDF -->
      <div id="tab-pdf" class="tab-content">
        <div class="upload-zone" onclick="document.getElementById('pdf_file').click()">
          <div class="upload-icon">&#8679;</div>
          <p>Click to select a PDF job posting</p>
          <p class="hint" style="margin-top:6px">Max 5MB &middot; Text-based PDFs only (not scanned images)</p>
          <input type="file" id="pdf_file" name="pdf_file" accept=".pdf" style="display:none"
            onchange="showFile(this)"/>
        </div>
        <p id="fname" class="hint" style="margin-top:8px;text-align:center"></p>
      </div>

      <button type="submit" class="btn">Analyze Job Posting &rarr;</button>
    </form>
  </div>

  <!-- Tips -->
  <div class="tips-card">
    <div class="tips-title">How to spot a fake job posting</div>
    <div class="tips-grid">
      <div class="tip">
        <div class="tip-icon">&#128176;</div>
        <div class="tip-title">Unrealistic salary</div>
        <div class="tip-text">Salaries far above market rate for the role or experience level are a major red flag.</div>
      </div>
      <div class="tip">
        <div class="tip-icon">&#128241;</div>
        <div class="tip-title">Personal contact info</div>
        <div class="tip-text">Legitimate companies use official email domains, not Gmail or WhatsApp for hiring.</div>
      </div>
      <div class="tip">
        <div class="tip-icon">&#9889;</div>
        <div class="tip-title">Urgency language</div>
        <div class="tip-text">"Apply now", "limited seats", "urgent hiring" — pressure tactics are common in scams.</div>
      </div>
      <div class="tip">
        <div class="tip-icon">&#128203;</div>
        <div class="tip-title">Vague requirements</div>
        <div class="tip-text">"No experience needed", "anyone can apply" — real jobs have specific skill requirements.</div>
      </div>
      <div class="tip">
        <div class="tip-icon">&#127968;</div>
        <div class="tip-title">Work from home bait</div>
        <div class="tip-text">Scam postings heavily promote easy remote work with guaranteed high income.</div>
      </div>
      <div class="tip">
        <div class="tip-icon">&#128269;</div>
        <div class="tip-title">No company info</div>
        <div class="tip-text">Real companies include their name, website, and office location in job postings.</div>
      </div>
    </div>
  </div>

</div>

<script>
function switchTab(name, btn) {
  ['text','url','pdf'].forEach(t => {
    document.getElementById('tab-'+t).classList.remove('active');
  });
  document.getElementById('tab-'+name).classList.add('active');
  document.getElementById('itype').value = name;
  document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
}

function updateCount() {
  const txt   = document.getElementById('job_text').value.trim();
  const words = txt ? txt.split(/\s+/).length : 0;
  const chars = txt.length;
  document.getElementById('charcount').textContent = words + ' words · ' + chars + ' chars';
}

function showFile(input) {
  document.getElementById('fname').textContent =
    input.files[0] ? 'Selected: ' + input.files[0].name : '';
}

function validate() {
  const type = document.getElementById('itype').value;
  if (type === 'text') {
    const txt = document.getElementById('job_text').value.trim();
    if (txt.length < 50) {
      alert('Please paste a complete job description (at least 50 characters).');
      return false;
    }
  } else if (type === 'url') {
    const url = document.getElementById('url_input').value.trim();
    if (!url.startsWith('http')) {
      alert('Please enter a valid URL starting with http:// or https://');
      return false;
    }
  }
  return true;
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
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#0f0f0f;color:#e8e8e8;padding:28px 20px;min-height:100vh}
.wrapper{max-width:820px;margin:0 auto}
.topbar{display:flex;justify-content:space-between;align-items:center;margin-bottom:24px}
.topbar h1{font-size:17px;font-weight:600;color:#fff}
.back{font-size:13px;color:#555;text-decoration:none;border:1px solid #222;padding:6px 14px;border-radius:8px;transition:all 0.2s}
.back:hover{color:#fff;border-color:#444}

/* Verdict */
.verdict{border-radius:16px;padding:28px 28px 24px;margin-bottom:18px;display:grid;grid-template-columns:1fr auto;gap:20px;align-items:center}
.verdict.safe{background:#071207;border:1px solid #1a3a1a}
.verdict.suspicious{background:#130f00;border:1px solid #332800}
.verdict.high-risk{background:#130404;border:1px solid #331010}
.verdict-left{}
.verdict-tag{font-size:11px;font-weight:600;letter-spacing:1px;text-transform:uppercase;margin-bottom:8px}
.safe .verdict-tag{color:#166534}
.suspicious .verdict-tag{color:#92400e}
.high-risk .verdict-tag{color:#7f1d1d}
.verdict-label{font-size:36px;font-weight:800;letter-spacing:-1px;margin-bottom:6px}
.safe .verdict-label{color:#4ade80}
.suspicious .verdict-label{color:#fbbf24}
.high-risk .verdict-label{color:#f87171}
.verdict-meta{font-size:13px;color:#555}
.verdict-meta span{color:#888;margin-right:16px}
.chart-wrap{width:130px;height:130px;flex-shrink:0}

/* Cards */
.card{background:#161616;border:1px solid #1e1e1e;border-radius:14px;padding:22px;margin-bottom:16px}
.card-title{font-size:11px;font-weight:700;color:#555;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:18px}
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:16px}

/* Score bars */
.score-row{display:flex;align-items:center;gap:12px;margin-bottom:11px}
.score-name{width:160px;font-size:13px;color:#888;flex-shrink:0}
.score-track{flex:1;background:#0f0f0f;border-radius:4px;height:7px;overflow:hidden}
.score-fill{height:7px;border-radius:4px;transition:width 1s ease}
.score-fill.low{background:#4ade80}
.score-fill.mid{background:#fbbf24}
.score-fill.high{background:#f87171}
.score-pct{width:44px;text-align:right;font-size:12px;color:#555}

/* Chips */
.chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}
.chip{font-size:12px;padding:4px 11px;border-radius:20px;font-weight:500}
.chip-red{background:#1a0808;border:1px solid #2e1515;color:#f87171}
.chip-amber{background:#130f00;border:1px solid #2e2200;color:#fbbf24}
.chip-gray{background:#161616;border:1px solid #222;color:#777}
.chip-green{background:#071207;border:1px solid #1a3a1a;color:#4ade80}

/* SHAP */
.shap-box{background:#0f0f0f;border:1px solid #1a1a1a;border-radius:10px;padding:16px;font-size:14px;line-height:2.3;margin-bottom:16px}
.legend{display:flex;gap:18px;font-size:12px;color:#555;margin-bottom:12px;flex-wrap:wrap}
.leg-dot{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:middle}

/* Word bars */
.word-title{font-size:11px;font-weight:700;letter-spacing:0.6px;text-transform:uppercase;margin-bottom:10px}
.word-title.red{color:#f87171}
.word-title.green{color:#4ade80}
.word-row{display:flex;align-items:center;gap:8px;margin-bottom:7px}
.word-name{width:96px;font-size:12px;font-family:monospace;color:#888;flex-shrink:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.word-track{flex:1;background:#0f0f0f;border-radius:3px;height:5px}
.word-bar-r{height:5px;border-radius:3px;background:#f87171}
.word-bar-g{height:5px;border-radius:3px;background:#4ade80}
.word-val{font-size:11px;color:#444;width:44px;text-align:right}

/* Conf boxes */
.conf-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
.conf-box{background:#0f0f0f;border:1px solid #1a1a1a;border-radius:10px;padding:14px;text-align:center}
.conf-num{font-size:22px;font-weight:700;color:#fff;margin-bottom:3px}
.conf-lbl{font-size:11px;color:#444;text-transform:uppercase;letter-spacing:0.4px}

/* Flags */
.flag-row{display:flex;align-items:flex-start;gap:10px;padding:9px 0;border-bottom:1px solid #1a1a1a;font-size:13px;color:#888}
.flag-row:last-child{border-bottom:none}
.flag-dot{width:6px;height:6px;border-radius:50%;background:#f87171;margin-top:5px;flex-shrink:0}

/* Note box */
.note{background:#0f0f0f;border:1px solid #1a1a1a;border-radius:8px;padding:10px 14px;font-size:13px;color:#555;margin-top:12px;line-height:1.5}

@media(max-width:620px){
  .verdict{grid-template-columns:1fr}.chart-wrap{display:none}
  .two-col,.conf-grid{grid-template-columns:1fr}
  .score-name{width:120px}
}
</style>
</head>
<body>
<div class="wrapper">

  <div class="topbar">
    <h1>Fake Job Detector</h1>
    <a href="/" class="back">&larr; Analyze another</a>
  </div>

  <!-- Verdict banner -->
  <div class="verdict {{ prediction.risk_label | lower | replace(' ','-') }}">
    <div class="verdict-left">
      <div class="verdict-tag">Analysis complete</div>
      <div class="verdict-label">{{ prediction.risk_label }}</div>
      <div class="verdict-meta">
        <span>Risk score: {{ (prediction.final_score * 100)|round(1) }}%</span>
        <span>Source: {{ source }}</span>
      </div>
    </div>
    <div class="chart-wrap">
      <canvas id="donut"></canvas>
    </div>
  </div>

  <!-- Score breakdown + bar chart -->
  <div class="two-col">
    <div class="card">
      <div class="card-title">Score breakdown</div>

      {% set scores = [
        ("ML model", prediction.ml_score),
        ("Suspicious phrases", prediction.phrase_score),
        ("Salary anomaly", prediction.salary.risk_score),
        ("Contact red flags", contact.risk_score),
        ("Keyword analysis", keywords.risk_score),
      ] %}

      {% for name, val in scores %}
      {% set pct = (val * 100)|round %}
      <div class="score-row">
        <div class="score-name">{{ name }}</div>
        <div class="score-track">
          <div class="score-fill {% if pct < 35 %}low{% elif pct < 65 %}mid{% else %}high{% endif %}"
               style="width:{{ pct }}%"></div>
        </div>
        <div class="score-pct">{{ pct }}%</div>
      </div>
      {% endfor %}

      {% if prediction.salary.reason %}
      <div class="note">{{ prediction.salary.reason }}</div>
      {% endif %}
    </div>

    <div class="card">
      <div class="card-title">Signal radar</div>
      <canvas id="barChart" height="200"></canvas>
    </div>
  </div>

  <!-- Flags detected -->
  {% if prediction.suspicious_phrases or contact.flags %}
  <div class="card">
    <div class="card-title">Red flags detected</div>

    {% if prediction.suspicious_phrases %}
    <div style="margin-bottom:14px">
      <div style="font-size:12px;color:#f87171;font-weight:600;margin-bottom:7px">Suspicious phrases</div>
      <div class="chips">
        {% for p in prediction.suspicious_phrases %}
        <span class="chip chip-red">{{ p }}</span>
        {% endfor %}
      </div>
    </div>
    {% endif %}

    {% if contact.flags %}
    {% for flag in contact.flags %}
    <div class="flag-row"><div class="flag-dot"></div><div>{{ flag }}</div></div>
    {% endfor %}
    {% endif %}
  </div>
  {% endif %}

  <!-- Keyword analysis -->
  <div class="card">
    <div class="card-title">Keyword analysis</div>
    <div style="font-size:13px;color:#666;margin-bottom:12px">
      {{ keywords.total_words }} total words &middot;
      Most frequent: <span style="font-family:monospace;color:#aaa">"{{ keywords.top_word }}"</span>
      at <span style="color:{% if keywords.is_stuffed %}#f87171{% else %}#4ade80{% endif %}">{{ keywords.top_density }}%</span>
      {% if keywords.is_stuffed %}<span class="chip chip-red" style="margin-left:6px;font-size:11px">Keyword stuffing</span>{% endif %}
    </div>

    <div class="two-col" style="gap:12px">
      {% if keywords.urgency_keywords %}
      <div>
        <div style="font-size:11px;color:#555;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.4px">Urgency</div>
        <div class="chips">{% for w in keywords.urgency_keywords %}<span class="chip chip-red">{{ w }}</span>{% endfor %}</div>
      </div>
      {% endif %}
      {% if keywords.money_keywords %}
      <div>
        <div style="font-size:11px;color:#555;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.4px">Money</div>
        <div class="chips">{% for w in keywords.money_keywords %}<span class="chip chip-amber">{{ w }}</span>{% endfor %}</div>
      </div>
      {% endif %}
      {% if keywords.vague_keywords %}
      <div>
        <div style="font-size:11px;color:#555;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.4px">Vague</div>
        <div class="chips">{% for w in keywords.vague_keywords %}<span class="chip chip-gray">{{ w }}</span>{% endfor %}</div>
      </div>
      {% endif %}
    </div>
  </div>

  <!-- SHAP -->
  <div class="card">
    <div class="card-title">Word-level explanation (SHAP)</div>
    <div class="legend">
      <span><span class="leg-dot" style="background:rgba(248,113,113,0.7)"></span>Pushes toward fake</span>
      <span><span class="leg-dot" style="background:rgba(74,222,128,0.6)"></span>Pushes toward real</span>
      <span style="font-size:11px;color:#333">Hover any word to see its score</span>
    </div>
    <div class="shap-box">{{ shap_html | safe }}</div>

    <div class="two-col">
      {% if fake_words %}
      <div>
        <div class="word-title red">Top fake signals</div>
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
        <div class="word-title green">Top real signals</div>
        {% for word, val in real_words %}
        <div class="word-row">
          <div class="word-name" title="{{ word }}">{{ word }}</div>
          <div class="word-track"><div class="word-bar-g" style="width:{{ [((val|float)*100*-1)|round, 100]|min }}%"></div></div>
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
        <div class="conf-lbl">Ensemble</div>
      </div>
    </div>
  </div>

</div>

<script>
const score    = {{ (prediction.final_score * 100)|round(1) }};
const label    = "{{ prediction.risk_label }}";
const riskColor = label === "Safe" ? "#4ade80" : label === "Suspicious" ? "#fbbf24" : "#f87171";
const chartData = {{ chart_data | safe }};

// Donut chart
new Chart(document.getElementById("donut"), {
  type: "doughnut",
  data: {
    datasets: [{
      data: [score, 100 - score],
      backgroundColor: [riskColor, "#1a1a1a"],
      borderWidth: 0,
      borderRadius: 4,
    }]
  },
  options: {
    cutout: "74%",
    plugins: {
      legend: { display: false },
      tooltip: { enabled: false }
    },
    animation: { animateRotate: true, duration: 900 }
  },
  plugins: [{
    id: "centerText",
    afterDraw(chart) {
      const { ctx, chartArea: { width, height, left, top } } = chart;
      ctx.save();
      ctx.font = "700 22px -apple-system, sans-serif";
      ctx.fillStyle = riskColor;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(score + "%", left + width/2, top + height/2);
      ctx.restore();
    }
  }]
});

// Bar chart
new Chart(document.getElementById("barChart"), {
  type: "bar",
  data: {
    labels: chartData.labels,
    datasets: [{
      data: chartData.values,
      backgroundColor: chartData.values.map(v =>
        v < 35 ? "rgba(74,222,128,0.7)" : v < 65 ? "rgba(251,191,36,0.7)" : "rgba(248,113,113,0.7)"
      ),
      borderRadius: 5,
      borderSkipped: false,
    }]
  },
  options: {
    indexAxis: "y",
    scales: {
      x: {
        max: 100,
        grid: { color: "#1a1a1a" },
        ticks: { color: "#444", font: { size: 11 }, callback: v => v + "%" }
      },
      y: {
        grid: { display: false },
        ticks: { color: "#666", font: { size: 11 } }
      }
    },
    plugins: { legend: { display: false }, tooltip: {
      callbacks: { label: ctx => " " + ctx.raw + "%" }
    }},
    animation: { duration: 800 }
  }
});
</script>
</body>
</html>
'''

# ─────────────────────────────────────────────────────────────────────────────
to_write = {
    "utils/job_validator.py":  JOB_VALIDATOR,
    "app.py":                  APP_PY,
    "templates/loading.html":  LOADING_HTML,
    "templates/index.html":    INDEX_HTML,
    "templates/result.html":   RESULT_HTML,
}

for rel, content in to_write.items():
    full = os.path.join(BASE, rel)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(content)
    size = os.path.getsize(full)
    print(f"  OK  {rel:40s}  {size:6} bytes")

print("\nAll files written successfully!")
print("Run:  python app.py")