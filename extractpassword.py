import cv2
from paddleocr import PaddleOCR
import re
from typing import List,Tuple, Optional, Dict
import numpy as np
import google.generativeai as genai
import re
import json
from PIL import Image
from passportgemini import api_key

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


# Initialize OCR engine
ocr = PaddleOCR(use_angle_cls=True, lang='en')


def filter_caps_and_dates(strings: List[str]) -> List[str]:
    """Filters uppercase strings and valid DD/MM/YYYY dates."""
    date_pattern = re.compile(r'\b(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[0-2])/\d{4}\b')
    mixed_caps_pattern = re.compile(r'^.*\/([A-Z]+)$')  # capture group for CAPS after last /

    filtered = []
    for s in strings:
        if s.isupper():
            filtered.append(s)
        elif s in ['p','s','o','d']:
            filtered.append(s)
        elif date_pattern.fullmatch(s):
            filtered.append(s)
        else:
            match = mixed_caps_pattern.fullmatch(s)
            if match:
                filtered.append(match.group(1))  # only the CAPS after slash
    return filtered


def display_image(image, window_name="Image"):
    """Displays an image using OpenCV."""
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def resize_by_width(image, width=1024):
    h, w = image.shape[:2]
    aspect_ratio = h / w
    new_height = int(width * aspect_ratio)
    return cv2.resize(image, (width, new_height))

def preprocess_image(image_path: str, target_size=None) -> np.ndarray:
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
    date=False,
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
    if(len(extracted_values)<3 and date):
        extracted_values=[]
    return extracted_values

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


def extract_passport_details(image_path: str) -> Dict[str, str]:
    """Extracts key passport details from the given image path."""

    def get_next_valid_date(data: List[str], start_index: int) -> Optional[int]:
        """Finds the next index of a valid date starting from `start_index`."""
        for idx in range(start_index, len(data)):
            if date_pattern.fullmatch(data[idx]):
                return idx
        return None

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
    image,Flag = preprocess_image(image_path)
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
        for i in details.keys():
            if details[i]=="":
                details=gemini_fallback(image_path,details)
    if("REPUBLIC" in full_text and flag==1):
        details=extract_front_page(raw_texts,result)
    if("EMIGRATION" in full_text and flag==1):
        raw_texts=filter_caps_and_dates(raw_texts)
        details.update(extract_back_page(raw_texts,result))
    return details

print(extract_passport_details("Screenshot 2025-06-13 103636.png"))