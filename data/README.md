# 📊 Dataset Documentation: Real or Fake Job Postings

This directory contains the training dataset for the **VerifID** fake job and internship detection system.

---

### 📥 Source & Download Instructions

The model is trained on the Kaggle **Real or Fake: Fake JobPosting Prediction** dataset published by Shivam Bansal.

- **Kaggle Link:** [Real or Fake Job Prediction Dataset](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
- **Target File Path:** `data/fake_jobs.csv`

#### Download via Kaggle CLI:
```bash
kaggle datasets download -d shivamb/real-or-fake-fake-jobposting-prediction
unzip real-or-fake-fake-jobposting-prediction.zip -d data/
```
Ensure the extracted CSV is named or renamed to `fake_jobs.csv` inside the `data/` directory.

---

### 📋 Dataset Overview

- **Total Job Postings:** 17,880
- **Real Postings (Class 0):** 17,014 (~95.16%)
- **Fraudulent Postings (Class 1):** 866 (~4.84%)
- **Target Field:** `fraudulent` (0 = Real, 1 = Fake)
- **Key Text Columns Combined:**
  - `title`
  - `company_profile`
  - `description`
  - `requirements`
  - `benefits`

---

### ⚠️ Note on Version Control
Because `fake_jobs.csv` is ~50 MB, it is excluded from Git tracking via `.gitignore`.
