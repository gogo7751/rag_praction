import base64
from fastapi import HTTPException
from core.llm import llm
from langchain.schema import HumanMessage

def process_with_llm(text: str) -> str:
    try:
        instruction = f"請將以下 markdown 轉換為 JSON 格式, 請把行政區當成 key,請回應我 json 內容就好, 其他文字都請不要回傳：\n\n{text}"
        response = llm.generate([[HumanMessage(content=instruction)]])
        return response.generations[0][0].text.strip()
    except Exception as e:
        raise RuntimeError(f"LLM 處理失敗: {e}")


def process_image_with_llm(image_path: str) -> str:
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        instruction = (
            "以下是一張圖片的 url，請提取其內容並結構化為 markdown 格式：\n\n"
            f"data:image;base64,{base64_image}"
        )
        response = llm.generate([[HumanMessage(content=instruction)]])
        return response.generations[0][0].text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"透過 LLM 處理圖片數據時出錯: {e}")
    