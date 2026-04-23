import os, sys
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
