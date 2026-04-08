# test_shap.py  (run from project root)
from explainability.shap_explainer import get_top_shap_words

text = """
Software Engineer - Remote
Work from home opportunity. No experience needed.
Earn $5000 weekly guaranteed. Urgent hiring. Apply now.
Send CV on WhatsApp. Free training provided.
"""

result = get_top_shap_words(text)

print("Top FAKE-pushing words:")
for word, val in result["fake_words"][:5]:
    print(f"  {word:20s}  shap={val:.4f}")

print("\nTop REAL-pushing words:")
for word, val in result["real_words"][:5]:
    print(f"  {word:20s}  shap={val:.4f}")

print("\nHTML preview (first 300 chars):")
print(result["html"][:300])