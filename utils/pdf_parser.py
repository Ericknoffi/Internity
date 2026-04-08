import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fitz   # PyMuPDF
import os

MAX_PAGES = 10   # ignore PDFs longer than this (likely not a job posting)

def extract_text_from_pdf(file_path: str) -> dict:
    """
    Extracts text from a PDF file.
    Returns:
        success   (bool)
        text      (str)
        pages     (int)
        error     (str)  — only on failure
    """
    if not os.path.exists(file_path):
        return {"success": False, "error": "Uploaded file not found."}

    try:
        doc = fitz.open(file_path)
    except Exception as e:
        return {"success": False, "error": f"Could not open PDF: {str(e)}"}

    if len(doc) > MAX_PAGES:
        return {
            "success": False,
            "error": f"PDF has {len(doc)} pages — please upload a single job posting (max {MAX_PAGES} pages)."
        }

    full_text = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")   # plain text extraction
        full_text.append(text)

    doc.close()

    combined = " ".join(full_text).strip()
    combined = " ".join(combined.split())   # collapse extra whitespace

    if len(combined) < 100:
        return {
            "success": False,
            "error": "PDF appears to be scanned/image-based with no extractable text. Please paste the text manually."
        }

    return {
        "success": True,
        "text": combined[:8000],
        "pages": len(full_text)
    }


def save_uploaded_pdf(file_storage, upload_folder: str) -> str:
    """
    Saves a Flask file upload to disk safely.
    Returns the saved file path.
    """
    os.makedirs(upload_folder, exist_ok=True)
    filename = "uploaded_job.pdf"
    save_path = os.path.join(upload_folder, filename)
    file_storage.save(save_path)
    return save_path