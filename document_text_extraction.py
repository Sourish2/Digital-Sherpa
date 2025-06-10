import tempfile
from fastapi import UploadFile
import shutil
import os
import cv2
from paddleocr import PaddleOCR
import re
import pdfplumber
import itertools
from typing import List, Optional, Dict

# Initialize OCR once
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Constants
BLACKLIST_KEYWORDS = [
    "INCOME TAX", "GOVT. OF INDIA", "GOVERNMENT", "DEPARTMENT", "PERMANENT ACCOUNT",
    "SIGNATURE", "YOUR SIGNATURE HERE"
]

DOB_BLACKLIST_CONTEXT = ["Issue Date", "DATE", "date", "Date", "DATE OF ISSUE","Address","S/O","Father"]
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


# Document Extractors

def extract_pan_details(image_path):
    image = preprocess_image_resize(image_path, (512, 512))
    result = ocr.ocr(image, cls=True)[0]
    lines = [line[1][0].strip() for line in result if line[1][0].strip() and not is_blacklisted(line[1][0])]
    print(lines)
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
    texts = [line[1][0] for line in result]
    print(texts)
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
                    while((any(bad in name for bad in DOB_BLACKLIST_CONTEXT) or len(name)<=3) and j > 0):
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
            if "S/O" in line.upper() and len(line.split()) > 1:
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
    def get_next_valid_date(data: List[str], start_index: int) -> Optional[int]:
        for idx in range(start_index, len(data)):
            if date_pattern.fullmatch(data[idx]):
                return idx
        return None

    # OCR and preprocessing
    img = preprocess_image_resize(image_path, (512, 512))

    result = ocr.ocr(img, cls=True)[0]
    raw_texts = [line[1][0] for line in result]
    filtered_data = filter_caps_and_dates(raw_texts)
    # Initialize fields
    details = {
        "Type": "",
        "Country Code": "",
        "Passport Number": "",
        "Sex": "",
        "Nationality": "",
        "Date of Birth": "",
        "Place of Birth": "",
        "Place of Issue": "",
        "Date of Issue": "",
        "Date of Expiry": "",
        "Full Name": ""
    }

    i = 0

    # Extract 'Type'
    for i, line in enumerate(filtered_data):
        if len(line) == 1 and line not in ['M', 'F']:
            details["Type"] = line
            i = i + 1
            break

    # Country Code
    if i < len(filtered_data) and len(filtered_data[i]) == 3:
        details["Country Code"] = filtered_data[i]
        i += 1

    # Passport Number
    if i < len(filtered_data) and re.fullmatch(passport_no_pattern, filtered_data[i]) or contains_digit(
            filtered_data[i]):
        details["Passport Number"] = filtered_data[i]
        i += 1

    # Full Name, Nationality, Gender
    name_parts = []
    while i < len(filtered_data) and filtered_data[i] not in ['M', 'F']:
        name_parts.append(filtered_data[i])
        i += 1

    if i < len(filtered_data):
        details["Sex"] = filtered_data[i]
        details["Nationality"] = name_parts.pop() if name_parts else ""
        details["Full Name"] = ' '.join(name_parts)
        i += 1

    # Date of Birth
    if i < len(filtered_data) and date_pattern.fullmatch(filtered_data[i]):
        details["Date of Birth"] = filtered_data[i]
        i += 1

        # Look for Date of Issue and Date of Expiry
        doi_index = get_next_valid_date(filtered_data, i)
        if doi_index is not None:
            details["Date of Issue"] = filtered_data[doi_index]

            doe_index = get_next_valid_date(filtered_data, doi_index + 1)
            if doe_index is not None:
                details["Date of Expiry"] = filtered_data[doe_index]

            # Guess Place of Birth and Place of Issue based on gaps
            if doi_index - i == 2:
                details["Place of Birth"] = filtered_data[i]
                details["Place of Issue"] = filtered_data[i + 1]
    return details


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
    else:
        return {"error": f"Unsupported document type: {document_name}"}

print(extract_pan_details("C:/Users/souri/PycharmProjects/intern/WhatsApp Image 2025-06-10 at 13.09.21_44bc07fe.jpg"))