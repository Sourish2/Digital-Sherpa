def is_blacklisted(text):
    return any(keyword in text.upper() for keyword in BLACKLIST_KEYWORDS)

def contains_digit(text):
    return any(char.isdigit() for char in text)

def normalize_ifsc(code):
    code = code.upper().replace(" ", "")
    code = code[:4].replace('1', 'I').replace('0', 'O') + code[4:]
    if len(code) >= 5:
        code = code[:4] + '0' + code[5:]
    return code

def normalize_bank(bank_name, ifsc_code):
    if bank_name:
        if "icici" in bank_name.lower():
            return "ICICI Bank"
        return bank_name.strip()
    prefix = normalize_ifsc(ifsc_code)[:4] if ifsc_code else ""
    return BANK_CODE_MAP.get(prefix, None)

def filter_caps_and_dates(strings: List[str]) -> List[str]:
    return [s for s in strings if s.isupper() or date_pattern.fullmatch(s)]

def clean_kv(entry, keys):
    if not entry:
        return None, None
    keys = sorted(keys, key=len, reverse=True)
    entry = entry.split("\n")
    if len(entry) == 1:
        for key in keys:
            if key in entry[0]:
                return key, entry[0].replace(key, "").strip()
    else:
        words = " ".join(entry).split()
        for i in range(1, len(words) + 1):
            for combo in itertools.combinations(words, i):
                candidate_key = " ".join(combo)
                if candidate_key in keys:
                    remaining_words = words.copy()
                    for word in combo:
                        if word in remaining_words:
                            remaining_words.remove(word)
                    return candidate_key, " ".join(remaining_words).strip()
    return None, None

def parse_table(table, keys):
    result = {}
    for row in table:
        for cell in row:
            key, value = clean_kv(cell, keys)
            if key:
                result[key] = value
    return result

def preprocess_image_resize(image_path, target_size):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path '{image_path}' could not be loaded.")
    image = cv2.resize(image, target_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path '{image_path}' could not be loaded.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image
