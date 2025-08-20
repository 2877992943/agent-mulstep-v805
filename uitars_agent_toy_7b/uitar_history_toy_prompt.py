from PIL import Image
import prompts
from prompts import(
    UITARS_ACTION_SPACE,
    UITARS_CALL_USR_ACTION_SPACE,
    UITARS_USR_PROMPT_NOTHOUGHT,
    UITARS_USR_PROMPT_THOUGHT,
    UITARS_NORMAL_ACTION_SPACE
)
import base64
from pathlib import Path
from io import BytesIO
import re


newtoken1=1024

#
# def get_model_proces(model_name_or_path):
#     import torch
#     from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoModelForCausalLM
#     from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
#     # model_name_or_path = '/root/autodl-tmp/xiao/Qwen/Qwen2.5-VL-72B-Instruct-AWQ'
#     model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name_or_path,
#                                                                # torch_dtype="auto",
#                                                                device_map="auto",
#                                                                torch_dtype=torch.bfloat16,
#                                                                attn_implementation="flash_attention_2")
#     # default processer
#     processor = AutoProcessor.from_pretrained(model_name_or_path,
#                                               min_pixels=1 * 28 * 28,
#                                               max_pixels=3300 * 28 * 28,
#                                               )
#     return model, processor
# def infer(messages,processor,model):
#     # Preparation for inference
#     text = processor.apply_chat_template(
#         messages, tokenize=False, add_generation_prompt=True
#     )
#     image_inputs, video_inputs = process_vision_info(messages)
#     inputs = processor(
#         text=[text],
#         images=image_inputs,
#         videos=video_inputs,
#         padding=True,
#         return_tensors="pt",
#     )
#     inputs = inputs.to("cuda")
#
#     # Inference
#     generated_ids = model.generate(**inputs, max_new_tokens=newtoken1)
#     generated_ids_trimmed = [
#         out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
#     ]
#     output_text = processor.batch_decode(
#         generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
#     )
#     print(output_text)
#     torch.cuda.empty_cache()
#     return output_text
# def add_box_token(input_string):
#     # Step 1: Split the string into individual actions
#     if "Action: " in input_string and "start_box=" in input_string:
#         suffix = input_string.split("Action: ")[0] + "Action: "
#         actions = input_string.split("Action: ")[1:]
#         processed_actions = []
#         for action in actions:
#             action = action.strip()
#             # Step 2: Extract coordinates (start_box or end_box) using regex
#             coordinates = re.findall(r"(start_box|end_box)='\((\d+),\s*(\d+)\)'", action)
#
#             updated_action = action  # Start with the original action
#             for coord_type, x, y in coordinates:
#                 # Convert x and y to integers
#                 updated_action = updated_action.replace(f"{coord_type}='({x},{y})'",
#                                                         f"{coord_type}='<|box_start|>({x},{y})<|box_end|>'")
#             processed_actions.append(updated_action)
#
#         # Step 5: Reconstruct the final string
#         final_string = suffix + "\n\n".join(processed_actions)
#     else:
#         final_string = input_string
#     return final_string
# def pil_to_base64(image):
#     buffer = BytesIO()
#     image.save(buffer, format="PNG")  # 你可以改成 "JPEG" 等格式
#     return base64.b64encode(buffer.getvalue()).decode("utf-8")
#
#
# def image_to_base64(image_path):
#     ext = Path(image_path).suffix.lower()
#     mime_types = {
#         '.jpg': 'image/jpeg',
#         '.jpeg': 'image/jpeg',
#         '.png': 'image/png',
#         '.gif': 'image/gif',
#         '.webp': 'image/webp',
#         '.bmp': 'image/bmp',
#         '.tiff': 'image/tiff',
#         '.svg': 'image/svg+xml',
#     }
#     with open(image_path, "rb") as image_file:
#         binary_data = image_file.read()
#         base64_data = base64.b64encode(binary_data).decode("utf-8")
#     return f"data:{mime_types.get(ext, 'image/png')};base64,{base64_data}"
#
# def predict(messages,img_paths,history_responses):
#     #history_responses=[]
#     image_num = 0
#     history_n=0
#     images=img_paths
#     if len( history_responses) > 0:### hisImg,hisRespons,....,currentImg,
#         for history_idx, history_response in enumerate( history_responses):
#             # send at most history_n images to the model只要这么多  历史图片
#             if history_idx  > len( history_responses)-history_n:
#                 cur_image = images[image_num]
#                 #encoded_string = pil_to_base64(cur_image)
#                 messages.append({
#                     "role": "user",
#                     "content": [{"type": "image",
#                                  "image": cur_image,
#
#                                  }]
#                 })
#                 image_num += 1
#             #### 所有的历史 response都要
#             messages.append({
#                 "role": "assistant",
#                 "content": [{"type":"text",
#                              "text":add_box_token(history_response)}]
#             })
#         ############# 历史结束
#         cur_image = images[image_num]
#         #encoded_string = pil_to_base64(cur_image)
#         messages.append({
#             "role": "user",
#             "content": [{"type": "image",
#                          "image": cur_image,
#
#                          }]
#         })
#         image_num += 1
#         return messages
#
#     else:
#         cur_image = images[image_num]
#         #encoded_string = pil_to_base64(cur_image)
#         messages.append({
#             "role": "user",
#             "content": [{"type": "image",
#                          "image":cur_image,
#
#                          }]
#         })
#         image_num += 1
#         return messages


actspace="\nclick(start_box='[x1, y1, x2, y2]')\nleft_double(start_box='[x1, y1, x2, y2]')\nright_single(start_box='[x1, y1, x2, y2]')\ndrag(start_box='[x1, y1, x2, y2]', end_box='[x3, y3, x4, y4]')\nhotkey(key='')\ntype(content='') #If you want to submit your input, use \"\\n\" at the end of `content`.\nscroll(start_box='[x1, y1, x2, y2]', direction='down or up or right or left')\nwait() #Sleep for 5s and take a screenshot to check for any changes.\nfinished(content='xxx')"
actspace=prompts.UITARS_ACTION_SPACE
def get_prompt(instruction):
    prompt_template = UITARS_USR_PROMPT_THOUGHT
    user_prompt = prompt_template.format(
        instruction=instruction, action_space=actspace, language='中文'
    )
    return user_prompt
# def get_message(user_prompt):
#
#     return [
#         {
#             "role": "system",
#             "content": [{"type": "text", "text": "You are a helpful assistant."}]
#         },
#         {
#             "role": "user",
#             "content": [{"type": "text", "text": user_prompt}]
#         }
#     ]
# if __name__=='__main__':
#     if False:
#         model_name_or_path='/root/autodl-tmp/yr/download_model/uitars-1.5-7b'
#         #instruction = '设置水印字体'
#         instruction='自定义快捷键，ctrl+c为复制'
#         #p0='1740366878.png'
#         #p0='1740366864.png'
#         #imgpath=f'/root/autodl-tmp/yr/data/305_elem_annote/1step/screens/{p0}'
#         imgpath='./p4.png'
#         #img=[Image.open(BytesIO(imgpath))]
#
#         user_prompt=get_prompt(instruction)
#         messages = get_message(user_prompt)
#         historyResp=['Thought: 我正在设置界面的快捷键区域，看到"复制"选项旁边显示的是"Ctrl+Shift+C"。要完成任务，我需要点击这个选项来修改快捷键。这样就能把当前的快捷键组合改成ctrl+c了。\nAction: click(start_box=\'(477,137)\')']
#         messages=predict(messages,[imgpath],historyResp )
#         model, processor = get_model_proces(model_name_or_path)
#         infer(messages)
#         respons=''
#
#     if 1:
#         res='Thought: 我看到系统设置窗口已经打开了，现在要设置水印字体。在"水印文字参数设置"区域里，我注意到字体下拉框显示的是"微软雅黑"。为了完成任务，我需要点击这个字体下拉框来选择其他字体\n'
#         respons="Action: click(start_box=<bbox>500 585 500 585</bbox>)"
#         o=add_box_token(res+respons)
#         print(o)

