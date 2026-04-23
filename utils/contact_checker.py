import re

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
