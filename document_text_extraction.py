import tempfile
from fastapi import UploadFile
import shutil
import os
import cv2
from paddleocr import PaddleOCR
import google.generativeai as genai
import re
import pdfplumber
import itertools
from typing import List, Optional, Dict
from PIL import Image
import json
from passportgemini import api_key
import numpy as np


# Initialize OCR once
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Constants
BLACKLIST_KEYWORDS = [
    "INCOME TAX", "GOVT. OF INDIA", "GOVERNMENT", "DEPARTMENT", "PERMANENT ACCOUNT",
    "SIGNATURE", "YOUR SIGNATURE HERE"
]

DOB_BLACKLIST_CONTEXT = ["Issue Date", "DATE", "date", "Date", "DATE OF ISSUE", "Address", "S/O", "Father"]
IFSC_PATTERN = r'[A-Z10]{4}[0O][A-Z0-9]{4,6}'
ACC_PATTERN = r'\d{11,18}\b'
PIN_PATTERN = r'(?<=[A-Z\-])\s*\d{3}\s*\d{3}(?=[A-Z ,\.]|$)'
AADHAR_PATTERN = r'\b\d{4}\D{0,3}\d{4}\D{0,3}\d{4}\b'
DOB_PATTERN = r'\d{2}[-/.]\d{2}[-/.]\d{4}\b'
passport_no_pattern = r'^[A-PR-WY][0-9]{7}$'
date_pattern = re.compile(r'\b(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[0-2])/\d{4}\b')

BANK_CODE_MAP = {
    "ICIC": "ICICI Bank", "IOBA": "Indian Overseas Bank", "HDFC": "HDFC Bank",
    "SBIN": "State Bank of India", "AXIS": "Axis Bank", "PUNB": "Punjab National Bank",
    "UBIN": "Union Bank of India"
}


# Utils

def is_blacklisted(text):
    return any(keyword in text.upper() for keyword in BLACKLIST_KEYWORDS)


def contains_digit(text):
    return any(char.isdigit() for char in text)

def resize_by_width(image, width=1024):
    h, w = image.shape[:2]
    aspect_ratio = h / w
    new_height = int(width * aspect_ratio)
    return cv2.resize(image, (width, new_height))

def preprocess_passport_image(image_path: str, target_size=None) -> np.ndarray:
    """Loads and preprocesses the passport image.
    If the top 25% of the image has very little text, cuts the image in half vertically.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path '{image_path}' could not be loaded.")

    h, w = image.shape[:2]
    top_quarter = image[0:int(h * 0.25), :]

    # Run OCR on top 25% only
    result = ocr.ocr(top_quarter, cls=True)[0]

    # Filter out noise/false positives with confidence threshold
    high_conf_texts = [line for line in result if line[1][1] > 0.5]
    Flag=1

    # If very few (e.g., ≤ 2) confident detections, assume it's mostly blank
    if len(high_conf_texts) <= 2:
        print("Top of image is mostly empty — cropping image in half.")
        image = image[int(h * 0.5):, :]
        image=resize_by_width(image)
        Flag=0

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),Flag

def extract_front_page(raw_texts,result):
    passport_no_pattern = r'[A-PR-WX-Z][0-9]{7}'
    date_pattern = re.compile(r'\b(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[0-2])/\d{4}\b')

    filtered_data = filter_caps_and_dates(raw_texts)

    passport_no = re.search(passport_no_pattern, " ".join(raw_texts))
    if passport_no:
        passport_no = passport_no.group()
    # Initialize fields
    # Get multi-field values once
    dates = extract_multiple_by_prefix(result, "Date of", True)
    places = extract_multiple_by_prefix(result, "Place of")

    # Use safe access with default values ("" if missing)
    def get_item(lst, idx):
        return lst[idx] if idx < len(lst) else ""

    details = {
        "Type": extract_by_keyword(result, "Type") or "",
        "Country Code": extract_by_keyword(result, "Country Code") or "",
        "Passport No.": passport_no or "",
        "Sex": extract_by_keyword(result, "Sex") or "",
        "Nationality": extract_by_keyword(result, "nat") or "",
        "Date of Birth": get_item(dates, 0),
        "Date of Issue": get_item(dates, 1),
        "Date of Expiry": get_item(dates, 2) or extract_by_keyword(result, "Date of Exp") or "",
        "Place of Birth": get_item(places, 0),
        "Place of Issue": get_item(places, 1),
        "Given Name(s)": extract_by_keyword(result, "Gi", 60) or "",
    }
    # Extract 'Type'
    for i, line in enumerate(filtered_data):
        if len(line) == 1 and line.lower() in ['p', 's', 'o', 'd']:
            if details["Type"] == "" or len(details["Type"]) > 1:
                details["Type"] = line
            i = i + 1
            break

    # Country Code
    if i < len(filtered_data) and len(filtered_data[i]) == 3:
        if (details["Country Code"] == ""):
            details["Country Code"] = filtered_data[i]
        i += 1
    # Gender
    for i in filtered_data:
        if details["Sex"] == "":
            if i == "M":
                details["Sex"] = "M"
            elif i == "F":
                details["Sex"] = "F"
    # Dates
    dates = []
    date_index = []
    for i, line in enumerate(filtered_data):
        if re.fullmatch(date_pattern, line):
            dates.append(line)
            date_index.append(i)
    if (details["Date of Birth"] == ""):
        details["Date of Birth"] = get_item(dates, 0)
    if (details["Date of Issue"] == ""):
        details["Date of Issue"] = get_item(dates, 1)
    if (details["Date of Expiry"] == ""):
        details["Date of Expiry"] = get_item(dates, 2)
    if (len(date_index) == 3 and date_index[1] - date_index[0] == 4):
        if (details["Place of Birth"] == ""):
            details["Place of Birth"] = filtered_data[date_index[0] + 2]
        if (details["Place of Issue"] == ""):
            details["Place of Issue"] = filtered_data[date_index[0] + 3]
    if (len(date_index) == 3 and date_index[1] - date_index[0] == 3):
        if (details["Place of Birth"] == ""):
            details["Place of Birth"] = filtered_data[date_index[0] + 1]
        if (details["Place of Issue"] == ""):
            details["Place of Issue"] = filtered_data[date_index[0] + 2]
    return details

def extract_back_page(raw_data,result):
    address=""
    details={"Address": ""}
    for i, line in enumerate(raw_data):
        if "," in line or len(line.split())>3:
            address=" ".join([raw_data[i],raw_data[i+1],raw_data[i+2]])
            break
    details["Address"]=address
    return details



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

def gemini_fallback(image_path,details):
    try:
        # Configure Gemini
        genai.configure(api_key=api_key)

        # Load the image
        image = Image.open(image_path)

        # Use correct model name (no trailing dot)
        model = genai.GenerativeModel("models/gemini-1.5-flash-latest")

        # Prompt and image
        response = model.generate_content(
            ["""
                Extract the following fields from this passport image and return them in a JSON dictionary format:

                {
                  "Type": "",
                  "Country Code": "",
                  "Passport No.": "",
                  "Sex": "",
                  "Nationality": "",
                  "Date of Birth": "",
                  "Date of Issue": "",
                  "Date of Expiry": "",
                  "Place of Birth": "",
                  "Place of Issue": "",
                  "Given Name(s)": "",
                  "Address": ""
                }

                Only return the dictionary without explanation.
                """, image],
            generation_config={"temperature": 0.2}
        )

        # Extract the JSON dictionary using regex
        match = re.search(r"\{.*\}", response.text, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            return data
        else:
            print("❌ JSON not found in Gemini response.")
            return details

    except genai.types.generation_types.StopCandidateException as e:
        print(f"❌ Gemini stopped generation early: {e}")
        return details
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse JSON from response: {e}")
        return details
    except Exception as e:
        print(f"❌ Gemini API error: {e}")
        return details



def filter_caps_and_dates(strings: List[str]) -> List[str]:
    """Filters uppercase strings and valid DD/MM/YYYY dates."""
    date_pattern = re.compile(r'\b(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[0-2])/\d{4}\b')
    mixed_caps_pattern = re.compile(r'^.*\/([A-Z]+)$')  # capture group for CAPS after last /

    filtered = []
    for s in strings:
        if s.isupper():
            filtered.append(s)
        elif date_pattern.fullmatch(s):
            filtered.append(s)
        else:
            match = mixed_caps_pattern.fullmatch(s)
            if match:
                filtered.append(match.group(1))  # only the CAPS after slash
    return filtered

def extract_by_keyword(result, keyword: str, y_tolerance: int = 40, x_tolerance: int = 80) -> Optional[str]:
    """
    Search for a keyword and extract the first text box directly underneath it
    (within y_tolerance vertically and x_tolerance horizontally).
    """
    keyword = keyword.lower()
    candidates = []

    for i, line in enumerate(result):
        text = line[1][0].strip().lower()
        box = line[0]

        if keyword in text:
            # Keyword box center
            (x1, y1), _, (x2, y2), _ = box
            keyword_y = (y1 + y2) / 2
            keyword_x = (x1 + x2) / 2

            # Check boxes below and within horizontal alignment
            for other in result[i+1:]:
                other_text = other[1][0].strip()
                obox = other[0]
                (ox1, oy1), _, (ox2, oy2), _ = obox
                oy_center = (oy1 + oy2) / 2
                ox_center = (ox1 + ox2) / 2

                if 0 < (oy_center - keyword_y) < y_tolerance and abs(ox_center - keyword_x) < x_tolerance:
                    candidates.append(other_text)

            break  # Only use the first match

    return candidates[0] if candidates else None
def extract_multiple_by_prefix(
    result,
    keyword_prefix: str,
    y_tolerance: int = 40,
    x_tolerance: int = 80
) -> List[str]:
    """
    Finds all keyword labels starting with a given prefix (e.g., 'Date of')
    and returns a list of the first text box directly underneath each match.

    Example:
        extract_multiple_by_prefix(result, 'Date of')
        => ['02/10/2003', '20/04/2022', '19/04/2032']
    """
    keyword_prefix = keyword_prefix.lower()
    extracted_values = []

    for i, line in enumerate(result):
        label_text = line[1][0].strip()
        label_lower = label_text.lower()
        box = line[0]

        if keyword_prefix in label_lower:
            # Get center of keyword label box
            (x1, y1), _, (x2, y2), _ = box
            keyword_y = (y1 + y2) / 2
            keyword_x = (x1 + x2) / 2

            # Search for the closest aligned value box underneath
            for other in result[i+1:]:
                value_text = other[1][0].strip()
                obox = other[0]
                (ox1, oy1), _, (ox2, oy2), _ = obox
                oy_center = (oy1 + oy2) / 2
                ox_center = (ox1 + ox2) / 2

                if 0 < (oy_center - keyword_y) < y_tolerance and abs(ox_center - keyword_x) < x_tolerance:
                    extracted_values.append(value_text)
                    break  # Stop at first value under each keyword label
    return extracted_values
def extract_date_near_keywords(
    ocr_result,
    keywords,
    exclude_if_contains=None,
    max_vertical_gap=50,
    max_horizontal_gap=150
):
    DOB_PATTERN = r'(?:\d{2}[-/.](?:\d{2}|\w{3})[-/.]\d{2,4})'
    exclude_if_contains = exclude_if_contains or []
    matched_date = None

    for i, (box, (text, conf)) in enumerate(ocr_result):
        upper_text = text.upper()

        if any(ex.upper() in upper_text for ex in exclude_if_contains):
            continue

        if any(keyword.upper() in upper_text for keyword in keywords):
            date_match = re.search(DOB_PATTERN, upper_text)
            if date_match:
                return date_match.group()

            if i + 1 < len(ocr_result):
                next_box, (next_text, _) = ocr_result[i + 1]
                next_upper = next_text.upper()
                current_box_y_center = sum(pt[1] for pt in box) / 4
                next_box_y_center = sum(pt[1] for pt in next_box) / 4
                if abs(next_box_y_center - current_box_y_center) <= 20:
                    if not any(ex.upper() in next_upper for ex in exclude_if_contains):
                        next_date_match = re.search(DOB_PATTERN, next_upper)
                        if next_date_match:
                            return next_date_match.group()

            keyword_bottom_y = max(pt[1] for pt in box)
            keyword_center_x = sum(pt[0] for pt in box) / 4
            closest_y_diff = float('inf')

            for other_box, (other_text, _) in ocr_result:
                other_upper = other_text.upper()

                if any(ex.upper() in other_upper for ex in exclude_if_contains):
                    continue

                other_top_y = min(pt[1] for pt in other_box)
                other_center_x = sum(pt[0] for pt in other_box) / 4
                y_diff = other_top_y - keyword_bottom_y

                if 0 < y_diff < max_vertical_gap and abs(other_center_x - keyword_center_x) < max_horizontal_gap:
                    other_date = re.search(DOB_PATTERN, other_upper)
                    if other_date and y_diff < closest_y_diff:
                        matched_date = other_date.group()
                        closest_y_diff = y_diff

            if matched_date:
                return matched_date

    return None


def extract_account_number(lines):
    STATE_CODES = [
        'AN', 'AP', 'AR', 'AS', 'BR', 'CH', 'CG', 'DD', 'DL', 'DN', 'GA', 'GJ', 'HP',
        'HR', 'JH', 'JK', 'KA', 'KL', 'LA', 'LD', 'MH', 'ML', 'MN', 'MP', 'MZ',
        'NL', 'OD', 'PB', 'PY', 'RJ', 'SK', 'TN', 'TR', 'TS', 'UK', 'UP', 'WB'
    ]
    for line in lines:
        upper_line = line.upper().replace(" ", "").replace("-", "")
        for code in STATE_CODES:
            if upper_line.startswith("DLNO" + code) or upper_line.startswith(code):
                # Extract using regex to avoid false positives
                match = re.search(r'([A-Z]{2}-?\d{4}-?\d{7,})', line)
                if match:
                    return match.group().replace(" ", "").strip()
                return line.replace("DLNo", "").replace("DL NO", "").replace(":", "").strip()
    return ""



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


# Document Extractors

def extract_pan_details(image_path):
    image = preprocess_image_resize(image_path, (512, 512))
    result = ocr.ocr(image, cls=True)[0]
    lines = [line[1][0].strip() for line in result if line[1][0].strip() and not is_blacklisted(line[1][0])]
    full_text = ' '.join(lines).upper()
    pan = re.search(r'\b[A-Z]{5}[0-9]{4}[A-Z]\b', full_text)
    dob = re.search(DOB_PATTERN, full_text)
    next_line = ""
    next_line2 = ""

    candidate_name, father_name = None, None
    for i in range(len(lines) - 1):
        current = lines[i]
        if i + 1 < len(lines):
            next_line = lines[i + 1]
        if i + 2 < len(lines):
            next_line2 = lines[i + 2]
            # Match both lines to uppercase full names
        if re.fullmatch(r'[A-Z ]{6,}', current) and (re.fullmatch(r'[A-Z ]{6,}', next_line)):
            candidate_name = current
            father_name = next_line
            break
        elif re.fullmatch(r'[A-Z ]{6,}', current) and (re.fullmatch(r'[A-Z ]{6,}', next_line2)):
            candidate_name = current
            father_name = next_line2
            break

    return {
        "Candidate Name": candidate_name or "",
        "Father Name": father_name or "",
        "DOB": dob.group() if dob else "",
        "PAN Number": pan.group() if pan else ""
    }


def extract_aadhar_details(image_path):
    image = preprocess_image(image_path)
    result = ocr.ocr(image, cls=True)[0]

    horizontal_texts = []
    for line in result:
        box = line[0]
        x_coords = [pt[0] for pt in box]
        y_coords = [pt[1] for pt in box]

        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)

        if width > height:  # width-dominant => horizontal
            horizontal_texts.append(line[1][0])

    texts = horizontal_texts#[line[1][0] for line in result]
    full_text = " ".join(texts)

    name, dob, aadhar_number, gender, address = None, None, None, None, None

    for i, line in enumerate(texts):
        if re.search(DOB_PATTERN, line) or "YEAR OF BIRTH" in line.upper():
            context = " ".join(texts[max(0, i - 1): i + 2]).upper()
            if not any(bad in context for bad in DOB_BLACKLIST_CONTEXT) or "DOB" in line.upper():
                dob_match = re.search(DOB_PATTERN, line)
                if dob_match:
                    dob = dob_match.group(0)
                else:
                    # Try to find a 4-digit year in the current line
                    year_match = re.search(r'(19|20)\d{2}', line)
                    if year_match:
                        dob = year_match.group(0)
                if i > 0:
                    name = texts[i - 1].strip().replace("Name", '').replace(':', '')
                    j = i - 1
                    while(j > 0 and (any(bad in name.upper() for bad in DOB_BLACKLIST_CONTEXT) or len(name) <= 3 or contains_digit(name))):
                        j -= 1
                        name = texts[j].strip().replace("Name", '').replace(':', '')
                break

    aadhar_number = re.search(AADHAR_PATTERN, full_text)
    aadhar_number = aadhar_number.group(0) if aadhar_number else ""

    for line in texts:
        if "MALE" in line.upper(): gender = "MALE"
        if "FEMALE" in line.upper(): gender = "FEMALE"

    address_lines = []
    for i, line in enumerate(texts):
        if "ADDRESS" in line.upper():
            for j in range(i + 1, min(i + 5, len(texts))):
                if not re.search(DOB_PATTERN, texts[j]) and not re.search(AADHAR_PATTERN, texts[j]):
                    address_lines.append(texts[j].strip())
                else:
                    break
            break

    address = ", ".join(address_lines) if address_lines else None
    if address == None:
        for i, line in enumerate(texts):
            if ("S/O" in line.upper() or "W/O" in line.upper()) and len(line.replace(":"," ").split()) > 1:
                # ✅ Include current line first
                candidate = line
                if not re.search(r'\d{2}[-/.]\d{2}[-/.]\d{4}', candidate) and \
                        not re.search(AADHAR_PATTERN, candidate) and \
                        not re.search(r'\bVID\b|\bDOB\b', candidate, re.IGNORECASE):
                    address_lines.append(candidate.strip())

                # Then check next 3 lines
                for j in range(i + 1, min(i + 4, len(texts))):
                    candidate = texts[j]
                    if not re.search(r'\d{2}[-/.]\d{2}[-/.]\d{4}', candidate) and \
                            not re.search(AADHAR_PATTERN, candidate) and \
                            not re.search(r'\bVID\b|\bDOB\b', candidate, re.IGNORECASE):
                        address_lines.append(candidate.strip())
                    else:
                        break
                break

        address = ", ".join(address_lines) if address_lines else None

    return {
        "Name": name or "",
        "Gender": gender or "",
        "DOB": dob or "",
        "Aadhar Number": aadhar_number,
        "Address": address or ""
    }


def extract_cheque_details(image_path):
    image = preprocess_image_resize(image_path, (1024, 512))
    result = ocr.ocr(image, cls=True)[0]
    texts = [line[1][0].strip() for line in result if line[1][0].strip()]
    full_text = " ".join(texts).upper()

    bank_name, branch_name, address, name = None, None, None, None
    ifsc, acc = None, re.search(ACC_PATTERN, full_text)

    for i, line in enumerate(texts):
        if not bank_name and re.search(r'BANK\b', line.upper()):
            bank_name = line
            if "BRANCH" in line.upper() and not (len(line.split()) > 5):
                branch_name = line
        if re.search(PIN_PATTERN, line.replace(" ", "")):
            address = line
        if any(kw in line.lower() for kw in ["please", "sign", "above"]):
            for j in range(i - 1, -1, -1):
                if len(texts[j].split()) <= 3 and not contains_digit(texts[j]) and len(texts[j]) > 2:
                    name = texts[j]
                    break

    for match in re.findall(IFSC_PATTERN, full_text):
        if any(c.isdigit() for c in match) and "FOR" not in match.upper() or "BAR" in match.upper():
            ifsc = match
            break
    if not ifsc:
        ifsc_match = re.search(IFSC_PATTERN, full_text)
        ifsc = ifsc_match.group() if ifsc_match else ""
    if not address:
        image = cv2.resize(image, (1500, 512))
        result = ocr.ocr(image, cls=True)[0]
        texts = [line[1][0].strip() for line in result if len(line[1][0].strip()) > 0]
        for i, line in enumerate(texts):
            if re.search(PIN_PATTERN, line.replace(" ", "")):
                address = line
    return {
        "IFSC Code": normalize_ifsc(ifsc) or "",
        "Bank Name": normalize_bank(bank_name, ifsc) or "",
        "Branch Name": branch_name or "",
        "Address": address or "",
        "Account Holder Name": name or "",
        "Account Number": acc.group() if acc else ""
    }


def extract_cml(pdf_file, keys):
    details={}
    if os.path.exists(pdf_file):
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    details.update(parse_table(table, keys))
            return details


def extract_voter_details(image_path):
    image = preprocess_image(image_path)
    result = ocr.ocr(image, cls=True)[0]
    lines = [re.sub(r'epic', '', line[1][0], flags=re.IGNORECASE).strip() for line in result if
             re.sub(r'epic', '', line[1][0], flags=re.IGNORECASE).strip() and "Download Date" not in line[1][0]]
    details = {"Name": "", "Relation Name": "", "Gender": "", "Date of Birth": "", "Address": ""}
    full_text = " ".join(lines).upper()
    details["Date of Birth"] = re.search(DOB_PATTERN, full_text)
    if details["Date of Birth"]:
        details["Date of Birth"] = details["Date of Birth"].group()
    else:
        details["Date of Birth"] = ""
    candidates = []
    address_lines = []
    relation = ["FATHER", "MOTHER", "HUSBAND", "GUARDIAN", "WIFE"]

    for i, line in enumerate(lines):
        upper_line = line.upper()
        if "MALE" in upper_line or "SEXM" in upper_line.replace(" ", "").replace("/", ""):
            details["Gender"] = "Male"
        elif "FEMALE" in upper_line or "SEXF" in upper_line.replace(" ", "").replace("/", ""):
            details["Gender"] = "Female"
        if "ADDR" in upper_line and details["Address"] == "":
            if "ADDRESS" in upper_line:
                idx = upper_line.find("ADDRESS")
                first_line = line[idx + len("ADDRESS"):].replace(':', '')
            else:
                idx = upper_line.replace(':', ' ').find(" ")
                first_line = line[idx:].replace(':', '')
            address_lines.append(first_line)
            if (first_line == ''):
                for j in range(3):
                    address_lines.append(lines[i + j + 1])
            else:
                for j in range(2):
                    address_lines.append(lines[i + j + 1])
            details["Address"] = " ".join(address_lines)
        if ("NAME" in upper_line or any(rel in upper_line for rel in relation)) and len(
                line.split()) <= 4 and '&' not in line:
            candidates.append(line)
            if i + 1 < len(lines) and len(lines[i + 1]) > 1:
                candidates.append(lines[i + 1])
            elif (i + 2 < len(lines)):
                candidates.append(lines[i + 2])

    # Extract Name
    if len(candidates) >= 2:
        name_line = candidates[0]
        name_tokens = name_line.split()
        name = " ".join([token.replace("Name", "").replace("NAME", '').replace(':', '') for token in name_tokens])
        if not name:
            name = candidates[1]
        details["Name"] = name.replace(":", "").strip()
    # Extract Relation's Name
    if len(candidates) >= 4:
        for i, relation_line in enumerate(candidates):
            upper_line = relation_line.upper()
            if any(rel in upper_line for rel in relation):
                relation_tokens = relation_line.split()
                relation_name = " ".join([
                    token.replace("Name", "").replace("NAME", '').replace(':', '') for token in relation_tokens
                    if all(rel not in token.upper() for rel in relation)])
                if not relation_name.strip():
                    j = i + 1
                    while (any(rel in candidates[j].upper() for rel in relation)):
                        j += 1
                    relation_name = candidates[j]
                details["Relation Name"] = relation_name.replace(":", "").strip()
                break

    return details


def extract_passport_details(image_path: str) -> Dict[str, str]:
    """Extracts key passport details from the given image path."""

    details={"Type": "",
          "Country Code": "",
          "Passport No.": "",
          "Sex": "",
          "Nationality": "",
          "Date of Birth": "",
          "Date of Issue": "",
          "Date of Expiry": "",
          "Place of Birth": "",
          "Place of Issue": "",
          "Given Name(s)": "",
          "Address" : ""}
    #Patterns
    passport_no_pattern = r'[A-PR-WX-Z][0-9]{7}'
    date_pattern = re.compile(r'\b(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[0-2])/\d{4}\b')

    # OCR and preprocessing
    flag=1
    image,Flag = preprocess_passport_image(image_path)
    if(Flag==1):
        image=cv2.resize(image,(512,512))
    result = ocr.ocr(image, cls=True)[0]
    raw_texts = [line[1][0] for line in result]
    keywords=["REPUBLIC", "EMIGRATION"]
    full_text=" ".join(raw_texts)
    details={}
    if all(keyword in full_text for keyword in keywords):
        print("Keywords detected — slicing image vertically in half.")

        # Get image dimensions
        h, w = image.shape[:2]
        mid = w // 2

        # Slice vertically
        left_half = image[:, :mid]
        right_half = image[:, mid:]

        # Resize both
        left_half = cv2.resize(left_half, (1024,1024))
        right_half = cv2.resize(right_half, (512,512))
        result = ocr.ocr(left_half, cls=True)[0]
        raw_texts = [line[1][0] for line in result]
        details = extract_front_page(raw_texts, result)
        result = ocr.ocr(right_half, cls=True)[0]
        raw_texts = [line[1][0] for line in result]
        raw_texts=filter_caps_and_dates(raw_texts)
        details.update(extract_back_page(raw_texts,result))
        flag=0
        # for i in details.keys():
        #     if details[i]=="":
        #         details=gemini_fallback(image_path)
    if("REPUBLIC" in full_text and flag==1):
        details=extract_front_page(raw_texts,result)
    if("EMIGRATION" in full_text and flag==1):
        raw_texts=filter_caps_and_dates(raw_texts)
        details.update(extract_back_page(raw_texts,result))
    return details

def extract_driver_details(image_path):
    DOB_PATTERN = r'(?:\d{2}[-/.](?:\d{2}|\w{3})[-/.]\d{2,4})'
    image = preprocess_image_resize(image_path, (1024, 1024))
    result = ocr.ocr(image, cls=True)[0]
    lines = [line[1][0].strip() for line in result if line[1][0].strip()
             and all(blacklist not in line[1][0].replace(" ", "").upper()
                     for blacklist in ["DRIVINGLICENCE", "GOVERNMENT", "STATE"])]

    account_number = extract_account_number(lines)
    name = ""
    doi = extract_date_near_keywords(result, ["Issue", "Doi", "ID"], ["Valid"])
    dob = extract_date_near_keywords(result, ["DOB", "Date of Birth"])
    valid = ""
    relation = ""
    address = ""
    bg = ""

    for i, line in enumerate(lines):
        upper_line = line.upper()

        if "NAME" in upper_line:
            content = line.replace("Name", "").replace(":", "").strip()
            if content == "":
                name = lines[i + 1]
                if len(name.split()) == 1:
                    name = name + " " + lines[i + 2]
            else:
                name = content
                if len(name.split()) == 1:
                    name = name + " " + lines[i + 1]

        if "VALID" in upper_line and any(x in upper_line.replace(" ", "") for x in ["NONTRANSPORT", "NON-TRANSPORT", "NT"]):
            for j in range(i, len(lines)):
                valid_match = re.search(DOB_PATTERN, lines[j])
                if valid_match:
                    valid = valid_match.group()
                    break

        if any(s in line.replace("/", "").replace("I", "") for s in ["SDW", "SWD", "Son", "Daughter", "Wife"]) or "/o" in line:
            relation = line.replace("/", "").replace("SDW", '').replace("SWD", '').replace("SonDaughterWife", "").replace("of", "")
            if len(relation) > 1:
                relation = relation[relation.find(":") + 1:]
            else:
                relation = relation.replace(":", "")
            if relation.strip() == "":
                relation = lines[i + 1]

        if "ADD" in upper_line:
            address_lines = [line]
            for j in range(i + 1, min(i + 4, len(lines))):
                address_lines.append(lines[j].strip())
            address = ', '.join(address_lines)

        if any(bgtype in upper_line for bgtype in ["A+", "B+", "AB+", "O+", "A-", "B-", "AB-", "O-", "BG", "GROUP"]) and bg == "" and not contains_digit(line):
            plus_idx = line.find('+')
            minus_idx = line.find('-')
            idx = plus_idx if plus_idx != -1 else minus_idx
            if idx != -1:
                if line[idx - 2:idx - 1] == "AB":
                    bg = line[idx - 2:idx + 1].strip()
                else:
                    bg = line[idx - 1:idx + 1].strip()
            else:
                bg = line.replace("BG", "").replace("Group", "").replace("Blood", "").replace(":", "").strip()

    if valid == "":
        valid = extract_date_near_keywords(result, ["Valid"])

    return {
        "Account Number": account_number,
        "Name": name,
        "Date of Issue": doi,
        "Date of Birth": dob,
        "Valid for": valid,
        "Relation": relation,
        "Address": address,
        "Blood Group": bg
    }



def save_upload_file_tmp(upload_file: UploadFile) -> str:
    # Use __file__ instead of _file_
    tmp_dir = os.path.join(os.path.dirname(__file__), '..', 'tmp_file_storage')
    os.makedirs(tmp_dir, exist_ok=True)

    suffix = os.path.splitext(upload_file.filename)[-1]
    tmp_file_path = os.path.join(tmp_dir, f"upload_{next(tempfile._get_candidate_names())}{suffix}")

    # Save the uploaded file
    upload_file.file.seek(0)
    with open(tmp_file_path, "wb") as out_file:
        shutil.copyfileobj(upload_file.file, out_file)

    return os.path.abspath(tmp_file_path)


async def extract_kyc_document_data(image: UploadFile, document_name: str):
    tmp_path = save_upload_file_tmp(image)
    if document_name == "UIDAI":
        return extract_aadhar_details(tmp_path)
    elif document_name == "PAN":
        return extract_pan_details(tmp_path)
    elif document_name == "CHEQUE":
        return extract_cheque_details(tmp_path)
    elif document_name == "CML":
        keys = [
            "DP Id", "Client Id", "Sex", "DP Int Ref No", "A/c Status", "A/c Opening Dt", "Purchase Waiver",
            "BO Status", "BO Sub Status", "A/c Category", "Freeze Status", "Registered For Easi",
            "Nationality", "Stmt Cycle", "Occupation", "Closure Init By", "Account Closure Dt", "Registered For Easi ",
            "Registered For Easiest", "SMS Registered", "SMS Mobile No", "UID", "RBI Ref No", "RBI Approval Dt",
            "Mode Of Operation", "BSDA Flag", "RGESS Flag", "Pledge SI Flag", "Email D/L Flag", "Annual Report Flag",
            "First Holder Name", "Second Holder Name", "Third Holder Name", "First Holder PAN", "Second Holder PAN",
            "Third Holder PAN"
        ]
        return extract_cml(tmp_path, keys)

    elif document_name == "PASSPORT":
        return extract_passport_details(tmp_path)
    elif document_name == "VOTERID":
        return extract_voter_details(tmp_path)
    elif document_name == "LICENCE":
        return extract_driver_details(tmp_path)
    else:
        return {"error": f"Unsupported document type: {document_name}"}
