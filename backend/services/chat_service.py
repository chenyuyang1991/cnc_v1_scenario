from models.chat import ChatResponse, ChatHistory
from datetime import datetime
from typing import Dict, Any
import uuid

chat_sessions = {}

def process_chat_message(message: str, context: Dict[str, Any]) -> ChatResponse:
    message_id = f"MSG-{str(uuid.uuid4())[:8].upper()}"
    
    response_content = generate_ai_response(message, context)
    show_file_upload = should_show_file_upload(message, context)
    show_config = should_show_config(message, context)
    show_results = should_show_results(message, context)
    
    return ChatResponse(
        id=message_id,
        content=response_content,
        type="assistant",
        showFileUpload=show_file_upload,
        showConfig=show_config,
        showResults=show_results,
        timestamp=datetime.utcnow().isoformat() + "Z"
    )

def generate_ai_response(message: str, context: Dict[str, Any]) -> str:
    message_lower = message.lower()
    
    if any(keyword in message_lower for keyword in ["開始", "新專案", "開始優化"]):
        return """歡迎使用 CNC AI 優化器！我將協助您優化 CNC 程式。

首先，請上傳您的 CNC 程式檔案（支援 .nc, .cnc, .gcode 格式）或使用我們的範本開始。

上傳檔案後，我會分析您的程式並提供優化建議。"""
    
    elif any(keyword in message_lower for keyword in ["檔案", "上傳", "程式"]):
        return """請上傳您的 CNC 程式檔案。我支援以下格式：
- G-Code 檔案 (.nc, .cnc, .gcode)
- Excel 範本 (.xlsx)
- CSV 資料檔 (.csv)

您也可以下載我們的範本檔案來開始。"""
    
    elif any(keyword in message_lower for keyword in ["配置", "設定", "參數"]):
        return """很好！現在讓我們設定優化參數。請在下方配置區域中設定：

1. **機台參數** - 主軸轉速、進給速度
2. **材料設定** - 材料類型、硬度
3. **刀具配置** - 刀具類型、直徑
4. **優化目標** - 時間、品質、刀具壽命
5. **安全限制** - 最大轉速、進給限制

設定完成後點擊「執行優化」開始分析。"""
    
    elif any(keyword in message_lower for keyword in ["優化", "執行", "分析"]):
        return """優化分析完成！結果顯示：

✅ **加工時間減少 23%** (節省 10.4 分鐘)
✅ **刀具壽命延長 15%** (約多 150 件)
✅ **品質分數 98.5%** (表面粗糙度 Ra 0.8)
✅ **成本節省 $127** 每件

請查看下方的詳細結果分析，包含圖表、程式碼差異、模擬預覽和驗證報告。"""
    
    elif any(keyword in message_lower for keyword in ["迭代", "修改", "調整"]):
        return """我了解您想要進一步優化。基於當前結果，我建議：

1. **參數調整** - 微調主軸轉速至 3500 RPM
2. **材料優化** - 考慮使用更適合的材料等級
3. **刀具策略** - 調整刀具路徑以提高效率
4. **策略改進** - 優化切削策略

請選擇您想要調整的方向，我會為您生成新的迭代版本。"""
    
    else:
        return f"""我理解您的問題："{message}"

作為 CNC AI 優化專家，我可以協助您：
- 分析和優化 CNC 程式
- 提供加工參數建議
- 生成優化報告
- 進行模擬驗證

請告訴我您具體需要什麼協助，或者直接上傳您的程式檔案開始優化。"""

def should_show_file_upload(message: str, context: Dict[str, Any]) -> bool:
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in ["開始", "新專案", "檔案", "上傳", "程式"])

def should_show_config(message: str, context: Dict[str, Any]) -> bool:
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in ["配置", "設定", "參數", "執行優化"])

def should_show_results(message: str, context: Dict[str, Any]) -> bool:
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in ["優化", "執行", "分析", "結果"])

def get_chat_history(session_id: str) -> ChatHistory:
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []
    
    return ChatHistory(
        session_id=session_id,
        messages=chat_sessions[session_id]
    )
