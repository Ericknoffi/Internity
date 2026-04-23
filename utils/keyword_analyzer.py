import re
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
