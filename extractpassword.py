import cv2
from paddleocr import PaddleOCR
import re
from typing import List, Optional, Dict
import numpy as np

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


def preprocess_image(image_path: str, target_size) -> np.ndarray:
    """Loads and preprocesses the passport image."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path '{image_path}' could not be loaded.")
    image = cv2.resize(image, target_size)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
def extract_passport_details(image_path: str) -> Dict[str, str]:
    """Extracts key passport details from the given image path."""

    def get_next_valid_date(data: List[str], start_index: int) -> Optional[int]:
        """Finds the next index of a valid date starting from `start_index`."""
        for idx in range(start_index, len(data)):
            if date_pattern.fullmatch(data[idx]):
                return idx
        return None
    #Patterns
    passport_no_pattern = r'[A-PR-WY][0-9]{7}'
    date_pattern = re.compile(r'\b(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[0-2])/\d{4}\b')

    # OCR and preprocessing
    img = preprocess_image(image_path,(512,512))
    result = ocr.ocr(img, cls=True)[0]
    raw_texts = [line[1][0] for line in result]
    filtered_data = filter_caps_and_dates(raw_texts)

    passport_no=re.search(passport_no_pattern," ".join(raw_texts))
    if passport_no:
        passport_no=passport_no.group()
    # Initialize fields
    # Get multi-field values once
    dates = extract_multiple_by_prefix(result, "Date of")
    places = extract_multiple_by_prefix(result, "Place of")

    # Use safe access with default values ("" if missing)
    def get_item(lst, idx):
        return lst[idx] if idx < len(lst) else ""

    details = {
        "Type": extract_by_keyword(result, "Type") or "",
        "Country Code": extract_by_keyword(result, "Country Code") or "",
        "Passport Number": passport_no or "",
        "Sex": extract_by_keyword(result, "Sex") or "",
        "Nationality": extract_by_keyword(result, "Nation") or "",
        "Date of Birth": get_item(dates, 0),
        "Date of Issue": get_item(dates, 1),
        "Date of Expiry": get_item(dates, 2) or extract_by_keyword(result, "Date of Exp") or "",
        "Place of Birth": get_item(places, 0),
        "Place of Issue": get_item(places, 1),
        "Name": extract_by_keyword(result, "Name") or ""
    }
     # Extract 'Type'
    for i, line in enumerate(filtered_data):
        if len(line) == 1 and line not in ['M', 'F']:
            if details["Type"] == "":
               details["Type"] = line
            i=i+1
            break

    # Country Code
    if i < len(filtered_data) and len(filtered_data[i]) == 3:
        if(details["Country Code"]==""):
            details["Country Code"] = filtered_data[i]
        i += 1
    #Gender
    for i in filtered_data:
        if details["Sex"]=="":
            if i=="M":
                details["Sex"]="M"
            elif i=="F":
                details["Sex"]="F"
    #Dates
    dates=[]
    date_index=[]
    print(filtered_data)
    for i,line in enumerate(filtered_data):
        if re.fullmatch(date_pattern,line):
            dates.append(line)
            date_index.append(i)
    if(details["Date of Birth"]==""):
        details["Date of Birth"]=get_item(dates,0)
    if (details["Date of Issue"] == ""):
        details["Date of Issue"] = get_item(dates, 1)
    if (details["Date of Expiry"] == ""):
        details["Date of Expiry"] = get_item(dates, 2)
    if(len(date_index)==3 and date_index[1]-date_index[0]==4):
        if (details["Place of Birth"] == ""):
            details["Place of Birth"]=filtered_data[date_index[0]+2]
        if (details["Place of Issue"] == ""):
            details["Place of Issue"] = filtered_data[date_index[0] + 3]
    if (len(date_index) == 3 and date_index[1] - date_index[0] == 3):
        if (details["Place of Birth"] == ""):
            details["Place of Birth"] = filtered_data[date_index[0] + 1]
        if (details["Place of Issue"] == ""):
            details["Place of Issue"] = filtered_data[date_index[0] + 2]

    return details

    # # Passport Number
    # if i < len(filtered_data) and re.fullmatch(passport_no_pattern, filtered_data[i]):
    #     details["Passport Number"] = filtered_data[i]
    #     i += 1
    #
    # # Full Name, Nationality, Gender
    # name_parts = []
    # while i < len(filtered_data) and filtered_data[i] not in ['M', 'F']:
    #     name_parts.append(filtered_data[i])
    #     i += 1
    #
    # if i < len(filtered_data):
    #     details["Sex"] = filtered_data[i]
    #     details["Nationality"] = name_parts.pop() if name_parts else ""
    #     details["Full Name"] = ' '.join(name_parts)
    #     i += 1
    #
    # # Date of Birth
    # if i < len(filtered_data) and date_pattern.fullmatch(filtered_data[i]):
    #     details["Date of Birth"] = filtered_data[i]
    #     i += 1
    #
    #     # Look for Date of Issue and Date of Expiry
    #     doi_index = get_next_valid_date(filtered_data, i)
    #     if doi_index is not None:
    #         details["Date of Issue"] = filtered_data[doi_index]
    #
    #         doe_index = get_next_valid_date(filtered_data, doi_index + 1)
    #         if doe_index is not None:
    #             details["Date of Expiry"] = filtered_data[doe_index]
    #
    #         # Guess Place of Birth and Place of Issue based on gaps
    #         if doi_index - i == 2:
    #             details["Place of Birth"] = filtered_data[i]
    #             details["Place of Issue"] = filtered_data[i + 1]
    # return details


print(extract_passport_details("C:/Users/souri/PycharmProjects/intern/Screenshot 2025-06-03 165441.png"))