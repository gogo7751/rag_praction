import json
from fastapi import APIRouter, HTTPException, UploadFile, Form
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Union
from core.agent import agent_with_memory
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from services.llm_process import process_with_llm, process_image_with_llm
from core.vectorstore import vectorstore
from services.ocr import extract_text_from_image

router = APIRouter()

class TextContent(BaseModel):
    type: str = "text"
    text: str

class ImageUrlContent(BaseModel):
    type: str = "image_url"
    image_url: dict

class MessageContent(BaseModel):
    role: str
    content: Union[str, List[Union[TextContent, ImageUrlContent]], dict]

class QuestionInput(BaseModel):
    model: str
    message: List[MessageContent]
    stream: bool

def process_content(messages: List[dict]) -> str:
    """
    處理 message 列表，提取每條消息的 content 並組合為上下文。
    """
    processed_texts = []
    for message in messages:
        if "content" in message and isinstance(message["content"], str):
            processed_texts.append(message["content"])
    return "\n".join(processed_texts)

@router.post("/upload-image/")
async def upload_image(file: UploadFile, use_ocr: bool = Form(True)):
    try:
        image_path = f"temp/{file.filename}"
        with open(image_path, "wb") as buffer:
            buffer.write(await file.read())

        if use_ocr:
            extracted_text = extract_text_from_image(image_path=image_path)
            if not extracted_text:
                return JSONResponse(content={"message": "無法從圖片中提取文字。"}, status_code=400)
            process_with_llm(extracted_text)
        else:
            structured_data = process_image_with_llm(image_path=image_path)

        vectorstore.add_texts([structured_data], metadatas={"source": file.filename})
        return {"message": "處理成功", "structured_data": structured_data}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@router.post("/ask-question/", response_class=PlainTextResponse)
async def ask_question(body: dict):
    try:
        # 提取並處理輸入
        messages = body.get("message", [])
        if not messages:
            raise ValueError("message 字段不可為空")
        combined_input = process_content(messages)

        async def stream():
            print(f"DEBUG: Combined Input: {combined_input}")

            # 調用 Agent
            response = agent_with_memory.invoke(combined_input)
            print(f"DEBUG: Raw Response: {response}")

            # 提取 output 中的內容
            if isinstance(response, dict) and "output" in response:
                output = response["output"]
                print(f"DEBUG: Extracted Output: {output}")

                # 嘗試解析 output 為 JSON
                try:
                    # 如果 output 是有效的 JSON 格式
                    output_data = json.loads(output)
                    if isinstance(output_data, dict) and "action_input" in output_data:
                        action_input = output_data["action_input"]
                        print(f"DEBUG: Extracted Action Input: {action_input}")
                        yield action_input  # 返回 action_input 值
                    else:
                        # 如果是 JSON，但不包含 action_input
                        yield output
                except json.JSONDecodeError:
                    # 如果 output 不是 JSON 格式，直接返回原始字串
                    print("DEBUG: Output is not JSON, returning as plain text")
                    yield output
            else:
                raise ValueError("Response 不包含 output 欄位")

        return StreamingResponse(stream(), media_type="text/plain")

    except Exception as e:
        print(f"ERROR: Outer exception: {e}")
        raise HTTPException(status_code=500, detail=f"處理用戶問題時出錯: {e}")
