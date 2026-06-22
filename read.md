# KYC Document Extraction API

A FastAPI-based OCR service for extracting structured information from Indian KYC documents using PaddleOCR and Gemini AI fallback.

## Supported Documents

| Document Type    | Code     |
| ---------------- | -------- |
| Aadhaar Card     | UIDAI    |
| PAN Card         | PAN      |
| Passport         | PASSPORT |
| Driving Licence  | LICENCE  |
| Voter ID         | VOTERID  |
| Cancelled Cheque | CHEQUE   |
| CML Statement    | CML      |

---

## Features

* OCR-powered document parsing using PaddleOCR
* Passport front/back page extraction
* Aadhaar information extraction
* PAN card details extraction
* Driving Licence details extraction
* Voter ID details extraction
* Cheque information extraction
* CML PDF parsing
* Gemini AI fallback for difficult passport extractions
* FastAPI REST API
* Swagger API documentation

---

## Tech Stack

* Python 3.10+
* FastAPI
* PaddleOCR
* OpenCV
* NumPy
* PDFPlumber
* Google Gemini API
* Pillow

---

## Project Structure

```text
DigitalSherpa/
│
├── main.py
├── routes.py
├── document_text_extraction.py
├── geminikey.py
├── requirements.txt
│
├── tmp_file_storage/
│
└── README.md
```

---

## Installation

### Clone Repository

```bash
git clone <repository-url>
cd DigitalSherpa
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

Linux / Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Gemini Configuration

Create a file named:

```text
geminikey.py
```

Add your Gemini API key:

```python
api_key = "YOUR_GEMINI_API_KEY"
```

---

## Running the Application

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

Server:

```text
http://127.0.0.1:8000
```

Swagger Documentation:

```text
http://127.0.0.1:8000/docs
```

ReDoc Documentation:

```text
http://127.0.0.1:8000/redoc
```

---

## API Endpoint

### Extract KYC Data

**POST**

```http
/api/extract
```

### Request

Form Data:

| Parameter     | Type       | Required |
| ------------- | ---------- | -------- |
| file          | UploadFile | Yes      |
| document_name | String     | Yes      |

Example:

```text
document_name = PASSPORT
```

### Supported Values

```text
UIDAI
PAN
PASSPORT
LICENCE
VOTERID
CHEQUE
CML
```

---

## Example Using Python

```python
import requests

url = "http://localhost:8000/extract"

files = {
    "file": open("passport.jpg", "rb")
}

data = {
    "document_name": "PASSPORT"
}

response = requests.post(
    url,
    files=files,
    data=data
)

print(response.json())
```

---

## Sample Response

```json
{
  "success": true,
  "document_type": "PASSPORT",
  "data": {
    "Passport No.": "X1234567",
    "Given Name(s)": "JOHN DOE",
    "Nationality": "INDIAN",
    "Date of Birth": "01/01/2000",
    "Date of Expiry": "01/01/2030"
  }
}
```

---

## Extracted Fields

### Aadhaar

* Name
* Gender
* Date of Birth
* Aadhaar Number
* Address

### PAN

* Candidate Name
* Father Name
* PAN Number
* Date of Birth

### Passport

* Type
* Country Code
* Passport Number
* Nationality
* Sex
* Date of Birth
* Date of Issue
* Date of Expiry
* Place of Birth
* Place of Issue
* Given Name(s)
* Address

### Driving Licence

* Licence Number
* Name
* Date of Birth
* Date of Issue
* Valid Till
* Relation
* Address
* Blood Group

### Voter ID

* Name
* Relation Name
* Gender
* Date of Birth
* Address

### Cheque

* IFSC Code
* Bank Name
* Branch Name
* Address
* Account Holder Name
* Account Number

### CML

* DP ID
* Client ID
* PAN Details
* Holder Information
* Account Metadata

---

## Security Notes

* Uploaded files should be deleted after processing.
* Do not commit Gemini API keys to source control.
* Restrict CORS origins before deploying to production.
* Store secrets using environment variables or AWS Secrets Manager.

---

## Future Improvements

* Async background processing
* Docker deployment
* AWS Lambda support
* RDS integration
* Document type auto-detection
* Batch uploads
* Confidence scoring
* MRZ extraction for passports

---

## License

Internal DigitalSherpa Project.
