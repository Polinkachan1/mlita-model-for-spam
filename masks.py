import re

PHONES = re.compile(r'(?<!\w)\+?\d[\d \-]{4,14}\d(?!\w)')
URLS = re.compile(r'(?:https?://|www\.)\S+|\b[\w-]+\.(?:com|net|org|co\.uk|info)\b\S*', re.IGNORECASE)

def apply_masks(text):
    if not isinstance(text, str):
        return ""
    text = URLS.sub(" maskedurl ", text)      # URLs first, so digits inside them aren't hit
    text = PHONES.sub(" maskedphone ", text)
    return text