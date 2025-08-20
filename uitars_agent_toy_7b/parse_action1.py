import re
import json

# 输入的字符串
input_str = """
click(start_box='<|box_start|>(12，45)<|box_end|>') 
left_double(start_box='<|box_start|>(103，1002)<|box_end|>') 
right_single(start_box='<|box_start|>(103，5690)<|box_end|>')
drag(start_box='<|box_start|>(56，901)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
hotkey(key='')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(start_box='<|box_start|>(567，102)<|box_end|>', direction='down or up or right or left')
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.
"""
import re
from typing import Dict, Any



def parse_action7b(line: str) -> Dict[str, Any]:
    # 移除注释部分
    clean_line = re.sub(r'#.*$', '', line).strip()

    # 主正则表达式：匹配动作名称和参数部分
    action_pattern = r'^(\w+)\((.*)\)$'
    action_match = re.match(action_pattern, clean_line)

    if not action_match:
        raise ValueError(f"无法解析的行: [{line}]")

    action_name = action_match.group(1)
    params_str = action_match.group(2)

    # 初始化参数字典
    params = {}

    # 处理坐标参数（支持两种格式：带标记和不带标记）
    coord_pattern = r"(start_box|end_box)='(?:<\|box_start\|>)?\((\d+)\s*,\s*(\d+)\)(?:<\|box_end\|>)?'"
    for match in re.finditer(coord_pattern, params_str):
        param_name = match.group(1)
        x, y = int(match.group(2)), int(match.group(3))
        params[param_name] = [x, y]

    # 处理其他键值参数
    kv_pattern = r"(\w+)=['\"](.*?)['\"]"
    for match in re.finditer(kv_pattern, params_str):
        key, value = match.group(1), match.group(2)
        if key not in params:  # 避免覆盖已解析的坐标参数
            params[key] = value

    # 特殊处理无参数的wait()
    if action_name == 'wait' and not params_str:
        params = {}

    return {
        'action': action_name,
        'params': params
    }

if __name__=='__main__':
    # 测试用例
    test_cases = [
        "click(start_box='<|box_start|>(12,45)<|box_end|>')",
        "left_double(start_box='<|box_start|>(103,1002)<|box_end|>')",
        "right_single(start_box='<|box_start|>(103,5690)<|box_end|>')",
        "drag(start_box='<|box_start|>(56,901)<|box_end|>', end_box='<|box_start|>(300,400)<|box_end|>')",
        "hotkey(key='')",
        "type(content='')",
        "scroll(start_box='<|box_start|>(567,102)<|box_end|>', direction='down or up or right or left')",
        "wait()",
        "finished(content='xxx')",
        "click(start_box='(1906,19)')"
    ][:]

    for case in test_cases:
        try:
            result = parse_action1(case)
            print(f"原始行: {case}")
            print(f"解析结果: {result}")
            print("-" * 50)
        except ValueError as e:
            print(f"错误: {e}")
            print("-" * 50)