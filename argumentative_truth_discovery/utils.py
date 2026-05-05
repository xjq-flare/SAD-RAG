import re
import json
from typing import Dict, Any, Tuple, Optional


def parse_think_and_json(response_text: str) -> Tuple[Optional[str], Dict[str, Any]]:

    reasoning_match = re.search(
        r'<think>(.*?)</think>',
        response_text,
        re.DOTALL
    )
    reasoning = reasoning_match.group(1).strip() if reasoning_match else None

    json_start = response_text.find('{')
    json_end = response_text.rfind('}') + 1
    
    if json_start >= 0 and json_end > json_start:
        json_str = response_text[json_start:json_end]
        json_data = json.loads(json_str)
    else:
        raise ValueError(f"无法找到 JSON 内容：{response_text}")

    if reasoning is None and "thought_process" in json_data:
        reasoning = json_data['thought_process']
    
    return reasoning, json_data

class ReturnData:
    def __init__(self, success, result, model_output, message):
        self.success = success
        self.result = result
        self.model_output = model_output
        self.message = message
        
    @staticmethod
    def success(result):
        return ReturnData(True, result, None, "success")
    
    @staticmethod
    def error(model_output, message):
        return ReturnData(False, None, model_output, message)
    
    def __repr__(self):
        return f"ReturnData(success={self.success}, result={self.result}, model_output={self.model_output}, message={self.message})"

