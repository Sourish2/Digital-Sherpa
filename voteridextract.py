import cv2
from paddleocr import PaddleOCR
import re
import numpy as np

# Initialize OCR engine
ocr = PaddleOCR(use_angle_cls=True, lang='en')

def preprocess_voter_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path '{image_path}' could not be loaded.")
        # Convert to gray
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply dilation and erosion to remove some noise
    #kernel = np.ones((2,2), np.uint8)
    # print(kernel)
    #image = cv2.dilate(image, kernel, iterations=1)
    #image = cv2.erode(image, kernel, iterations=1)
    return image

def display_image(image, window_name="Image"):

    # Display the image in a window
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

DOB_PATTERN = r'\d{2}[-/.]\d{2}[-/.]\d{4}\b'
EPIC_PATTERN=r'\b[A-Z]{3}[0-9]{7}\b'

def extract_voter_details(image_path):
    image = preprocess_voter_image(image_path)
    result = ocr.ocr(image, cls=True)[0]
    lines = [re.sub(r'epic', '', line[1][0], flags=re.IGNORECASE).strip()for line in result  if re.sub(r'epic', '', line[1][0], flags=re.IGNORECASE).strip() and "Download Date" not in line[1][0]]
    details = {"Name": "", "Relation Name": "", "Gender":"", "Date of Birth":"", "Address":"", "Epic Number":""}
    full_text = " ".join(lines).upper()
    details["Date of Birth"]=re.search(DOB_PATTERN,full_text)
    details["Epic Number"] = re.search(EPIC_PATTERN, full_text)
    if details["Date of Birth"]:
        details["Date of Birth"]=details["Date of Birth"].group()
    else:
        details["Date of Birth"] = ""
    if details["Epic Number"]:
        details["Epic Number"]=details["Epic Number"].group()
    else:
        details["Epic Number"] = ""
    candidates = []
    address_lines=[]
    relation=["FATHER","MOTHER","HUSBAND","GUARDIAN","WIFE"]

    for i, line in enumerate(lines):
        upper_line = line.upper()
        if "MALE" in upper_line or "SEXM" in upper_line.replace(" ","").replace("/",""):
            details["Gender"]="Male"
        elif "FEMALE" in upper_line or "SEXF" in upper_line.replace(" ","").replace("/",""):
            details["Gender"]="Female"
        if "ADDR" in upper_line and details["Address"]=="":
            if "ADDRESS" in upper_line:
                idx=upper_line.find("ADDRESS")
                first_line = line[idx+len("ADDRESS"):].replace(':', '')
            else:
                idx=upper_line.replace(':',' ').find(" ")
                first_line=line[idx:].replace(':','')
            address_lines.append(first_line)
            if(first_line==''):
                for j in range(3):
                    address_lines.append(lines[i+j+1])
            else:
                for j in range(2):
                    address_lines.append(lines[i+j+1])
            details["Address"]=" ".join(address_lines)
        if ("NAME" in upper_line or any(rel in upper_line for rel in relation)) and len(line.split())<=4 and '&' not in line:
            candidates.append(line)
            if i + 1 < len(lines) and len(lines[i+1])>1:
                candidates.append(lines[i + 1])
            elif(i + 2 < len(lines)):
                candidates.append(lines[i + 2])

    # Extract Name
    if len(candidates) >= 2:
        name_line = candidates[0]
        name_tokens = name_line.split()
        name = " ".join([token.replace("Name","").replace("NAME",'').replace(':','') for token in name_tokens])
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
                    token.replace("Name","").replace("NAME",'').replace(':','') for token in relation_tokens
                    if all(rel not in token.upper() for rel in relation)])
                if not relation_name.strip():
                    j=i+1
                    while(any(rel in candidates[j].upper() for rel in relation)):
                        j+=1
                    relation_name = candidates[j]
                details["Relation Name"] = relation_name.replace(":", "").strip()
                break

    return details


print(extract_voter_details("C:/Users/souri/PycharmProjects/Intern/Screenshot 2025-06-02 104428.png"))