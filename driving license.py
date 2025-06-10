# import cv2
# import re
# from paddleocr import PaddleOCR
#
# ocr = PaddleOCR(use_angle_cls=True, lang='en')
#
# DOB_PATTERN = r'(?:\d{2}[-/.](?:\d{2}|\w{3})[-/.]\d{2,4})'
#
#
# def extract_date_near_keywords(
#     ocr_result,
#     keywords,
#     exclude_if_contains=None,
#     max_vertical_gap=50,
#     max_horizontal_gap=150
# ):
#     exclude_if_contains = exclude_if_contains or []
#     matched_date = None
#
#     for i, (box, (text, conf)) in enumerate(ocr_result):
#         upper_text = text.upper()
#
#         if any(ex.upper() in upper_text for ex in exclude_if_contains):
#             continue
#
#         if any(keyword.upper() in upper_text for keyword in keywords):
#             # 1. Check on the same line
#             date_match = re.search(DOB_PATTERN, upper_text)
#             if date_match:
#                 return date_match.group()
#
#             # 2. Check the next horizontal line in OCR result (if exists)
#             if i + 1 < len(ocr_result):
#                 next_box, (next_text, _) = ocr_result[i + 1]
#                 next_upper = next_text.upper()
#
#                 # Get vertical midpoints to measure vertical alignment
#                 current_box_y_center = sum(pt[1] for pt in box) / 4
#                 next_box_y_center = sum(pt[1] for pt in next_box) / 4
#
#                 # Set a reasonable vertical alignment threshold (in pixels)
#                 vertical_alignment_threshold = 20
#
#                 # Check: Next line is not diagonally below, but horizontally close
#                 if abs(next_box_y_center - current_box_y_center) <= vertical_alignment_threshold:
#                     if not any(ex.upper() in next_upper for ex in exclude_if_contains):
#                         next_date_match = re.search(DOB_PATTERN, next_upper)
#                         if next_date_match:
#                             return next_date_match.group()
#
#             # 3. Vertical scan below keyword line (existing logic)
#             keyword_bottom_y = max(pt[1] for pt in box)
#             keyword_center_x = sum(pt[0] for pt in box) / 4
#             closest_y_diff = float('inf')
#
#             for other_box, (other_text, _) in ocr_result:
#                 other_upper = other_text.upper()
#
#                 if any(ex.upper() in other_upper for ex in exclude_if_contains):
#                     continue
#
#                 other_top_y = min(pt[1] for pt in other_box)
#                 other_center_x = sum(pt[0] for pt in other_box) / 4
#                 y_diff = other_top_y - keyword_bottom_y
#
#                 if 0 < y_diff < max_vertical_gap and abs(other_center_x - keyword_center_x) < max_horizontal_gap:
#                     other_date = re.search(DOB_PATTERN, other_upper)
#                     if other_date and y_diff < closest_y_diff:
#                         matched_date = other_date.group()
#                         closest_y_diff = y_diff
#
#             if matched_date:
#                 return matched_date
#
#     return None
# STATE_CODES = [
#     'AN', 'AP', 'AR', 'AS', 'BR', 'CH', 'CG', 'DD', 'DL', 'DN', 'GA', 'GJ', 'HP',
#     'HR', 'JH', 'JK', 'KA', 'KL', 'LA', 'LD', 'MH', 'ML', 'MN', 'MP', 'MZ',
#     'NL', 'OD', 'PB', 'PY', 'RJ', 'SK', 'TN', 'TR', 'TS', 'UK', 'UP', 'WB'
# ]
#
# def contains_digit(text):
#     return any(char.isdigit() for char in text)
#
#
# def preprocess_image_resize(image_path, target_size):
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError(f"Image at path '{image_path}' could not be loaded.")
#     image = cv2.resize(image, target_size)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     return image
#
# def extract_driver_details(image_path):
#     image = preprocess_image_resize(image_path, (1024,1024))
#     result = ocr.ocr(image, cls=True)[0]
#     lines = [line[1][0].strip() for line in result if line[1][0].strip() and all(blacklist not in line[1][0].replace(" ","").upper() for blacklist in ["DRIVINGLICENCE","GOVERNMENT","STATE"])]
#     account_number=""
#     name=""
#     doi=extract_date_near_keywords(result,["Issue","Doi","ID"],["Valid"])
#     dob=extract_date_near_keywords(result,["DOB","Date of Birth"])
#     valid=""
#     relation=""
#     address=""
#     bg=""
#     Flag=True
#     print(lines)
#     for i, line in enumerate(lines):
#         upper_line = line.upper()
#         line_has_no = "NO" in upper_line
#         valid_candidate = (
#                 any(code in upper_line for code in STATE_CODES) and
#                 Flag and
#                 contains_digit(line) and
#                 len(line.split()) < 2
#         )
#         if line_has_no and valid_candidate:
#             account_number = re.sub(r'DL\s?No[:\-]?', '', line, flags=re.IGNORECASE).strip()
#             Flag = False
#         elif not line_has_no and valid_candidate:
#             account_number = re.sub(r'DL\s?No[:\-]?', '', line, flags=re.IGNORECASE).strip()
#             Flag = False
#         if "Name" in line:
#             if line.replace("Name","").replace(":","")=="":
#                 name=lines[i+1]
#                 if len(name.split()) == 1:
#                     name = name + " " + lines[i + 2]
#             else:
#                 name=line.replace("Name","").replace(":","")
#                 if len(name.split()) == 1:
#                     name = name + " " + lines[i + 1]
#         if "VALID" in line.upper() and any(notransport in line.upper().replace(" ","") for notransport in ["NONTRANSPORT","NON-TRANSPORT","NT"]):
#             for j in range(i, len(lines)):
#                 valid = re.search(DOB_PATTERN, lines[j])
#                 if (valid):
#                     valid = valid.group()
#                     break
#         if any(s in line.replace("/","").replace("I","") for s in ["SDW","SWD","Son","Daughter","Wife"]) or "/o" in line:
#             relation = line.replace("/", "").replace("SDW", '').replace("SWD", '').replace("SonDaughterWife","").replace("of", "")
#             if(len(relation)>1):
#                 relation=relation[relation.find(":")+1:]
#             else:
#                 relation=relation.replace(":","")
#             if relation=="" or relation==" ":
#                 relation=lines[i+1]
#         if "ADD" in line.upper():
#             address_lines=[]
#             address_lines.append(line)
#             for j in range(i+1, min(i+4, len(lines))):
#                 address_lines.append(lines[j].strip())
#             address=', '.join(address_lines)
#         if any(kw in line.upper() for kw in ["A+","B+","AB+","O+","A-","B-","AB-","O-","BG","GROUP"]) and bg=="" and not contains_digit(line):
#             plus_idx = line.find('+')
#             minus_idx = line.find('-')
#             if plus_idx != -1:
#                 idx = plus_idx
#             elif minus_idx != -1:
#                 idx = minus_idx
#             else:
#                 idx = -1
#             if idx != -1:
#                 if(line[idx-2:idx-1]=="AB"):
#                     bg = line[idx - 2:idx + 1].strip()
#                 else:
#                     bg = line[idx - 1:idx + 1].strip()
#             else:
#                 bg = line.replace("BG", "").replace("Group", "").replace("Blood", "").replace(":", "").strip()
#     if(valid==""):
#         valid=extract_date_near_keywords(result,["Valid"])
#     return{"Account Number": account_number,
#     "Name": name,
#     "Date of Issue": doi,
#     "Date of Birth": dob,
#     "Valid for": valid,
#     "Relation": relation,
#     "Address":address,
#     "Blood Group": bg}

import cv2
import re
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='en')

DOB_PATTERN = r'(?:\d{2}[-/.](?:\d{2}|\w{3})[-/.]\d{2,4})'

STATE_CODES = [
    'AN', 'AP', 'AR', 'AS', 'BR', 'CH', 'CG', 'DD', 'DL', 'DN', 'GA', 'GJ', 'HP',
    'HR', 'JH', 'JK', 'KA', 'KL', 'LA', 'LD', 'MH', 'ML', 'MN', 'MP', 'MZ',
    'NL', 'OD', 'PB', 'PY', 'RJ', 'SK', 'TN', 'TR', 'TS', 'UK', 'UP', 'WB'
]


def contains_digit(text):
    return any(char.isdigit() for char in text)


def preprocess_image_resize(image_path, target_size):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path '{image_path}' could not be loaded.")
    image = cv2.resize(image, target_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def extract_date_near_keywords(
    ocr_result,
    keywords,
    exclude_if_contains=None,
    max_vertical_gap=50,
    max_horizontal_gap=150
):
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


def extract_driver_details(image_path):
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


# Example usage

print(extract_driver_details("C:/Users/souri/PycharmProjects/intern/Screenshot 2025-06-05 110529.png"))