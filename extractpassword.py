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
    return [s for s in strings if s.isupper() or date_pattern.fullmatch(s)]


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


def extract_passport_details(image_path: str) -> Dict[str, str]:
    """Extracts key passport details from the given image path."""

    def get_next_valid_date(data: List[str], start_index: int) -> Optional[int]:
        """Finds the next index of a valid date starting from `start_index`."""
        for idx in range(start_index, len(data)):
            if date_pattern.fullmatch(data[idx]):
                return idx
        return None

    # Patterns
    passport_no_pattern = r'^[A-PR-WY][0-9]{7}$'
    date_pattern = re.compile(r'\b(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[0-2])/\d{4}\b')

    # OCR and preprocessing
    img = preprocess_image(image_path,(512,512))
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

    # Extract 'Type'
    for i, line in enumerate(filtered_data):
        if len(line) == 1 and line not in ['M', 'F']:
            details["Type"] = line
            i=i+1
            break

    # Country Code
    if i < len(filtered_data) and len(filtered_data[i]) == 3:
        details["Country Code"] = filtered_data[i]
        i += 1

    # Passport Number
    if i < len(filtered_data) and re.fullmatch(passport_no_pattern, filtered_data[i]):
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


print(extract_passport_details("C:/Users/souri/PycharmProjects/intern/Screenshot 2025-05-28 155909.png"))