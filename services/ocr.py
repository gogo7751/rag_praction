from PIL import Image
from io import BytesIO
from fastapi import HTTPException
import pytesseract
import base64

def extract_text_from_image(image_path: str = None, base64_image: str = None) -> str:
    try:
        if base64_image:
            if "," in base64_image:
                base64_image = base64_image.split(",")[1] 
            image_data = base64.b64decode(base64_image)
            image = Image.open(BytesIO(image_data))
        elif image_path:
            image = Image.open(image_path)
        else:
            raise ValueError("必須提供 image_path 或 base64_image 之一")
        
        text = pytesseract.image_to_string(image, lang="chi_tra+eng")
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR 提取文字時出錯: {e}")
