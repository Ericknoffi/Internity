import os
from flask import Flask, render_template, request, redirect, url_for
from model.predict import predict
from explainability.shap_explainer import get_top_shap_words
from utils.scraper import scrape_job_url
from utils.pdf_parser import extract_text_from_pdf, save_uploaded_pdf

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB max PDF

ALLOWED_EXTENSIONS = {"pdf"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    text   = ""
    source = "Text input"
    error  = None

    input_type = request.form.get("input_type", "text")

    # ── 1. URL input ──────────────────────────────────────────────────────────
    if input_type == "url":
        url    = request.form.get("url", "").strip()
        result = scrape_job_url(url)
        if not result["success"]:
            return render_template("index.html", error=result["error"])
        text   = result["text"]
        source = f"Scraped from {result['source']}"

    # ── 2. PDF input ──────────────────────────────────────────────────────────
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

    # ── 3. Plain text input ───────────────────────────────────────────────────
    else:
        text = request.form.get("job_text", "").strip()
        if len(text) < 50:
            return render_template("index.html", error="Please enter at least 50 characters of job description.")
        source = "Pasted text"

    # ── 4. Run prediction ─────────────────────────────────────────────────────
    try:
        prediction = predict(text)
    except Exception as e:
        return render_template("index.html", error=f"Prediction error: {str(e)}")

    # ── 5. Run SHAP ───────────────────────────────────────────────────────────
    try:
        shap_result = get_top_shap_words(text)
    except Exception as e:
        shap_result = {"html": text, "fake_words": [], "real_words": []}

    return render_template(
        "result.html",
        source        = source,
        prediction    = prediction,
        shap_html     = shap_result["html"],
        fake_words    = shap_result["fake_words"][:8],
        real_words    = shap_result["real_words"][:8],
        raw_text      = text[:1000],
    )

if __name__ == "__main__":
    app.run(debug=True)