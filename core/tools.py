from langchain.tools import StructuredTool
from core.vectorstore import vectorstore

def get_commute_ratio(district: str) -> str:
    """
    用於查詢指定行政區的通勤比例，包括自行車和其他通勤方式。
    返回格式化的純文字字串。
    """
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        results = retriever.invoke(district)

        if not results:
            return f"{district} 沒有相關數據。"

        # 提取並格式化結果
        response_lines = []
        for doc in results:
            if hasattr(doc, "page_content") and doc.page_content.strip():
                content = doc.page_content.strip()
                metadata_desc = doc.metadata.get("description", "未知數據來源")
                response_lines.append(f"{district} 的通勤數據 ({metadata_desc}):\n{content}")

        # 合併為單一字串返回
        print("tool response:", "\n\n".join(response_lines))
        return "\n\n".join(response_lines)
    except Exception as e:
        return f"工具執行時發生錯誤: {str(e)}"
       
# 定義工具
t_get_commute_ratio = StructuredTool.from_function(
    get_commute_ratio,
    description="用於查詢指定行政區的通勤比例，包括自行車和其他通勤方式",
)

tools = [t_get_commute_ratio]