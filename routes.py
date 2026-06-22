from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from document_text_extraction import extract_kyc_document_data

router = APIRouter()


@router.post("/extract")
async def extract_document(
    file: UploadFile = File(...),
    document_name: str = Form(...)
):
    try:
        result = await extract_kyc_document_data(
            image=file,
            document_name=document_name.upper()
        )

        return {
            "success": True,
            "document_type": document_name.upper(),
            "data": result
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )