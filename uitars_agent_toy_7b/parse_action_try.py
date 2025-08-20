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
from parse_action1 import parse_action7b


def parsing_response_to_pyautogui_code(responses, image_height: int, image_width: int, input_swap: bool = True) -> str:
    '''
    对于 依赖模型定义的 actionType,attribute 写成 pyautogui
    将M模型的输出解析为OSWorld中的action，生成pyautogui代码字符串
    参数:
        response: 包含模型输出的字典，结构类似于：
        {
            "action_type": "hotkey",
            "action_inputs": {
                "hotkey": "v ctrl",
                "start_box": None,
                "end_box": None
            }
        }
    返回:
        生成的pyautogui代码字符串
    '''

    def escape_single_quotes(text):
        # 匹配未转义的单引号（不匹配 \\'）
        pattern = r"(?<!\\)'"
        return re.sub(pattern, r"\\'", text)

    pyautogui_code = f"import pyautogui\nimport time\n"

    image_width = 1
    image_height = 1

    if isinstance(responses, dict):
        responses = [responses]
    for response_id, response in enumerate(responses):  #### 遍历 一次预测中的  每一个 actiondict
        ##### 换成 action_type
        if 'action' in response:
            response['action_type'] = response['action']
        if not 'action_inputs' in response:
            response['action_inputs'] = response

        #####
        if "observation" in response:
            observation = response["observation"]
        else:
            observation = ""

        if "thought" in response:
            thought = response["thought"]
        else:
            thought = ""

        if response_id == 0:
            pyautogui_code += f"'''\nObservation:\n{observation}\n\nThought:\n{thought}\n'''\n"
        else:
            pyautogui_code += f"\ntime.sleep(1)\n"

        action_dict = response
        action_type = action_dict.get("action_type")
        action_inputs = action_dict.get("action_inputs", {})

        if action_type == "hotkey":
            # Parsing hotkey action
            if "key" in action_inputs:
                hotkey = action_inputs.get("key", "")
            else:
                hotkey = action_inputs.get("hotkey", "")

            if hotkey == "arrowleft":
                hotkey = "left"

            elif hotkey == "arrowright":
                hotkey = "right"

            elif hotkey == "arrowup":
                hotkey = "up"

            elif hotkey == "arrowdown":
                hotkey = "down"

            if hotkey:
                # Handle other hotkeys
                keys = hotkey.split()  # Split the keys by space
                convert_keys = []
                for key in keys:
                    if key == "space":
                        key = ' '
                    convert_keys.append(key)
                pyautogui_code += f"\npyautogui.hotkey({', '.join([repr(k) for k in convert_keys])})"

        elif action_type == "press":
            # Parsing press action
            if "key" in action_inputs:
                key_to_press = action_inputs.get("key", "")
            else:
                key_to_press = action_inputs.get("press", "")

            if hotkey == "arrowleft":
                hotkey = "left"

            elif hotkey == "arrowright":
                hotkey = "right"

            elif hotkey == "arrowup":
                hotkey = "up"

            elif hotkey == "arrowdown":
                hotkey = "down"

            elif hotkey == "space":
                hotkey = " "

            if key_to_press:
                # Simulate pressing a single key
                pyautogui_code += f"\npyautogui.press({repr(key_to_press)})"

        elif action_type == "keyup":
            key_to_up = action_inputs.get("key", "")
            pyautogui_code += f"\npyautogui.keyUp({repr(key_to_up)})"

        elif action_type == "keydown":
            key_to_down = action_inputs.get("key", "")
            pyautogui_code += f"\npyautogui.keyDown({repr(key_to_down)})"

        elif action_type == "type":
            # Parsing typing action using clipboard
            content = action_inputs.get("content", "")
            content = escape_single_quotes(content)
            stripped_content = content
            if content.endswith("\n") or content.endswith("\\n"):
                stripped_content = stripped_content.rstrip("\\n").rstrip("\n")
            if content:
                if input_swap:
                    pyautogui_code += f"\nimport pyperclip"
                    pyautogui_code += f"\npyperclip.copy('{stripped_content}')"
                    pyautogui_code += f"\npyautogui.hotkey('ctrl', 'v')"
                    pyautogui_code += f"\ntime.sleep(0.5)\n"
                    if content.endswith("\n") or content.endswith("\\n"):
                        pyautogui_code += f"\npyautogui.press('enter')"
                else:
                    pyautogui_code += f"\npyautogui.write('{stripped_content}', interval=0.1)"
                    pyautogui_code += f"\ntime.sleep(0.5)\n"
                    if content.endswith("\n") or content.endswith("\\n"):
                        pyautogui_code += f"\npyautogui.press('enter')"


        elif action_type in ["drag", "select"]:
            # Parsing drag or select action based on start and end_boxes
            start_box = action_inputs.get("start_box")
            end_box = action_inputs.get("end_box")
            if type(start_box) != str: start_box = str(start_box)
            if type(end_box) != str: end_box = str(end_box)
            if start_box and end_box:
                ### 这是起点
                startll = eval(start_box)
                if len(startll) == 4:
                    x1, y1, x2, y2 = eval(start_box)  # Assuming box is in [x1, y1, x2, y2]
                if len(startll) == 2:
                    x1, y1 = eval(start_box)
                    x2 = x1
                    y2 = y1
                #####
                sx = round(float((x1 + x2) / 2) * image_width, 3)
                sy = round(float((y1 + y2) / 2) * image_height, 3)

                ####end 这是终点
                endll = eval(end_box)
                if len(endll) == 4:
                    x1, y1, x2, y2 = eval(end_box)  # Assuming box is in [x1, y1, x2, y2]
                if len(endll) == 2:
                    x1, y1 = eval(end_box)
                    x2 = x1
                    y2 = y1
                #####
                ex = round(float((x1 + x2) / 2) * image_width, 3)
                ey = round(float((y1 + y2) / 2) * image_height, 3)
                ###
                sx = int(sx)
                sy = int(sy)
                ex = int(ex)
                ey = int(ey)
                # pyautogui_code += (
                #     f"\npyautogui.moveTo({sx}, {sy})\n"
                #     f"\npyautogui.dragTo({ex}, {ey}, duration=1.0)\n"
                # )
                pyautogui_code += (
                    f"\npyautogui.moveTo({sx}, {sy})\n"
                    f"\npyautogui.dragTo({ex}, {ey})\n"
                )

        elif action_type == "scroll":
            # Parsing scroll action
            start_box = action_inputs.get("start_box")
            if start_box:
                x1, y1, x2, y2 = eval(start_box)  # Assuming box is in [x1, y1, x2, y2]
                x = round(float((x1 + x2) / 2) * image_width, 3)
                y = round(float((y1 + y2) / 2) * image_height, 3)

                # # 先点对应区域，再滚动
                # pyautogui_code += f"\npyautogui.click({x}, {y}, button='left')"
            else:
                x = None
                y = None
            direction = action_inputs.get("direction", "")

            if x == None:
                if "up" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(5)"
                elif "down" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(-5)"
            else:
                if "up" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(5, x={x}, y={y})"
                elif "down" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(-5, x={x}, y={y})"

        elif action_type in ["click", "left_single", "left_double", "right_single", "hover"]:
            # Parsing mouse click actions
            start_box = action_inputs.get("start_box")
            start_box = str(start_box)
            if start_box:
                start_box = eval(start_box)
                if len(start_box) == 4:
                    x1, y1, x2, y2 = start_box  # Assuming box is in [x1, y1, x2, y2]
                elif len(start_box) == 2:
                    x1, y1 = start_box
                    x2 = x1
                    y2 = y1
                # x = round(float((x1 + x2) / 2) * image_width, 3)
                # y = round(float((y1 + y2) / 2) * image_height, 3)
                x = round(float((x1 + x2) / 2), 3)
                y = round(float((y1 + y2) / 2), 3)
                x = int(x);
                y = int(y)
                if action_type == "left_single" or action_type == "click":
                    pyautogui_code += f"\npyautogui.click({x}, {y}, button='left')"
                elif action_type == "left_double":
                    pyautogui_code += f"\npyautogui.doubleClick({x}, {y}, button='left')"
                elif action_type == "right_single":
                    pyautogui_code += f"\npyautogui.click({x}, {y}, button='right')"
                elif action_type == "hover":
                    pyautogui_code += f"\npyautogui.moveTo({x}, {y})"

        elif action_type in ["finished"]:
            pyautogui_code = f"DONE"
        elif action_type in ['wait']:
            pyautogui_code='\nimport time;time.sleep(3)'

        else:
            pyautogui_code += f"\n# Unrecognized action type: {action_type}"

    return pyautogui_code

def parse_action_output(output_text): ### 解析 thought和action, action在进一步解析
    # 提取Thought部分
    thought_match = re.search(r'Thought:(.*?)\nAction:', output_text, re.DOTALL)
    thought = thought_match.group(1).strip() if thought_match else ""

    # 提取Action部分
    action_match = re.search(r'Action:(.*?)(?:\n|$)', output_text, re.DOTALL)
    action_text = action_match.group(1).strip() if action_match else ""

    action_text=action_text.replace('<|diff_marker|>','').replace('<|im_end|>','')
    print('thought:',thought)
    print('parse act output:',action_text)
    print('output text',output_text)
    parsed_act=parse_action7b(action_text)
    print(parsed_act)
    #print(done)

    # 初始化结果字典
    result = {
        "thought": thought,
        "action": "",
        "key": None,
        "content": None,
        "start_box": None,
        "end_box": None,
        "direction": None
    }

    if 'drag' in parsed_act['action']:
        parsed_act['action']='drag'
    result['action']=parsed_act['action']
    params=parsed_act['params']
    if 'start_box' in params:
        result['start_box_int']=params['start_box']
        result['start_box']=params['start_box'] #[v/1000. for v in params['start_box']]
    if 'end_box' in params:
        result['end_box_int'] = params['end_box']
        result['end_box'] =  params['end_box']#[v / 1000. for v in params['end_box']]
    if 'direction' in params:
        result['direction']=params['direction']
    if 'content' in params:
        result['content']=params['content']
    if 'key' in params:
        result['key']=params['key']
    return json.dumps(result, ensure_ascii=False, indent=2)


if __name__=='__main__':
    t = 'Thought: 看到屏幕右上角那个关闭按钮了，就是那个"×"图标。既然要关闭当前应用，我只需要点击它就可以了。这个操作很简单直接，一步就能完成任务。\nAction: click(start_box=\'(1906,17)\')'
    parse_action_output(t)

