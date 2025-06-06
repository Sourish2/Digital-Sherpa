import cv2
import re
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='en')

DOB_PATTERN = r'\d{2}[-/.]\d{2}[-/.]\d{4}\b'


STATE_CODES = [
    'AN', 'AP', 'AR', 'AS', 'BR', 'CH', 'CG', 'DD', 'DL', 'DN', 'GA', 'GJ', 'HP',
    'HR', 'JH', 'JK', 'KA', 'KL', 'LA', 'LD', 'MH', 'ML', 'MN', 'MP', 'MZ',
    'NL', 'OD', 'PB', 'PY', 'RJ', 'SK', 'TN', 'TR', 'TS', 'UK', 'UP', 'WB'
]

def preprocess_image_resize(image_path, target_size):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path '{image_path}' could not be loaded.")
    image = cv2.resize(image, target_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def extract_driver_details(image_path):
    image = preprocess_image_resize(image_path, (1024,1024))
    result = ocr.ocr(image, cls=True)[0]
    lines = [line[1][0].strip() for line in result if line[1][0].strip() and all(blacklist not in line[1][0].replace(" ","").upper() for blacklist in ["DRIVINGLICENCE","GOVERNMENT","STATE"])]
    account_number=""
    name=""
    doi=""
    dob=""
    valid=""
    relation=""
    address=""
    bg=""
    Flag=True
    for i, line in enumerate(lines):
        if any(code in line for code in STATE_CODES) and Flag:
            account_number = re.sub(r'DL\s?No', '', line)
            Flag=False
        if "Name" in line:
            if line.replace("Name","").replace(":","")=="":
                name=lines[i+1]
                if len(name.split()) == 1:
                    name = name + " " + lines[i + 2]
            else:
                name=line.replace("Name","").replace(":","")
                if len(name.split()) == 1:
                    name = name + " " + lines[i + 1]
        if any(issue in line.upper() for issue in ["ISSUE","DOI","ID"]) and doi=="":
            for j in range(i, len(lines)):
                doi = re.search(DOB_PATTERN, lines[j])
                if (doi):
                    doi = doi.group()
                    break
        if "VALID" in line.upper():
            for j in range(i, len(lines)):
                valid = re.search(DOB_PATTERN, lines[j])
                if (valid):
                    valid = valid.group()
                    break
        if ("DOB" in line  or "Date of Birth" in line) and dob=="":
            for j in range(i, len(lines)):
                dob=re.search(DOB_PATTERN, lines[j])
                if (dob):
                    dob = dob.group()
                    break
        if any(s in line.replace("/","") for s in ["SDW","SWD"]):
            relation = line.replace("/", "").replace("SDW", '').replace("SWD", '').replace(':', '').replace("of", "")
            if relation=="" or relation==" ":
                relation=lines[i+1]
        if "ADD" in line.upper():
            address_lines=[]
            address_lines.append(line)
            for j in range(i+1, min(i+4, len(lines))):
                address_lines.append(lines[j].strip())
            address=', '.join(address_lines)
        if "BG" in line.upper() or "GROUP" in line.upper():
            bg=line.replace("BG","").replace(":","").replace("Group","").replace("Blood","").upper()
            if bg=="" or bg==" ":
                bg=lines[i+1].upper()
    return{"Account Number": account_number,
    "Name": name,
    "Date of Issue": doi,
    "Date of Birth": dob,
    "Valid for": valid,
    "Relation": relation,
    "Address":address,
    "Blood Group": bg}



print(extract_driver_details("Screenshot 2025-06-05 110529.png"))