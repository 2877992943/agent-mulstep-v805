import torch
import json
import sys,copy
sys.path.insert(0,'code805_2im/GUI-Actor-main_lora_616_uitarsFormat_II_2img')

from qwen_vl_utils import process_vision_info
from datasets import load_dataset
from transformers import AutoProcessor
#from gui_actor.constants import chat_template,guiact_action_system_message, grounding_system_message
from gui_actor.prompts_uitars import  get_prompt_uitars,get_prompt_uitars_noThought
from gui_actor.modeling_2img import Qwen2VLForConditionalGenerationWithPointer
from gui_actor.inference_2img import inference
import os
def get_data_process(model_name_or_path):
    return AutoProcessor.from_pretrained(model_name_or_path)
def get_model(model_name_or_path):
    return Qwen2VLForConditionalGenerationWithPointer.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="flash_attention_2"
    ).eval()
# def make_history_cell(h):
#     return {
#         "from": "gpt",
#         "value": h,  # "history1 thought1",
#         "loss_mask": 0
#     }
def make_image_cell():
    return {
        "from": "human",
        "value": "<image> ",
        "loss_mask": 0
    }
def get_imgbase64_content(encoded_string,min_pixels=256,max_pixels=3300*28*28):
    return {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": f"data:image/png;base64,{encoded_string}",
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,

            }]}
def make_dragEnd_cell():
    drag_prefix= '拖拽动作已经预测起点，现在预测终点'
    return {
        "from": "human",
        "value": drag_prefix,
        "loss_mask": 0
    }
def get_conversation_imgbase64(task,convs,png=None,encoded_string_img=None,base64flag=True):### 参考os-world agent 历史 处理
    """
    uitars
    message输入顺序： [系统提示词 + 任务] + [历史...] + 图
    convs提供顺序，
    其中内容,文本内容在里面，而图片内容在别处：  历史1 历史2...   图
    """
    #### 组装系统提示词+任务
    system_message_task = get_prompt_uitars(task)
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": system_message_task}]
        }]
    ######  历史1 历史2...  图
    for conv in convs:
        value=conv['value']
        lossmask=conv['loss_mask']
        if lossmask==0 and '<image>' not in value:# history
            history_response=value
            his={
                "role": "assistant",
                "content": [{"type": "text",
                             "text":  history_response}]
            }
            messages.append(his)
        if '<image>' in value:
            if not base64flag :
                #cur_image=os.path.join(inputImgdir,png)
                cur_image=png
                messages.append(
                    {
                    "role": "user",
                    "content": [{ "type": "image",
                                  "image": cur_image}]}
                )
            else:
                messages.append(get_imgbase64_content(encoded_string_img))
    return messages
def get_message2_conversation_imgbase64(task,convs,encoded_string_img_list):### 参考os-world agent 历史 处理
    """2 张图
    输入顺序： 系统提示词 + 任务 + 历史 + 图"""
    roles = {"human": "user", "gpt": "assistant", "system": "system"}
    if type(encoded_string_img_list)!=list:
        pngll=[encoded_string_img_list]
    #### 组装系统提示词+任务
    system_message_task = get_prompt_uitars(task)
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": system_message_task}]
        }]
    img_ix=0
    for conv in convs:
        value=conv['value']
        lossmask=conv['loss_mask']
        role=conv['role'] if 'role' in conv else conv['from']
        role=roles.get(role,role)#### 转换 role
        if lossmask==0 and '<image>' not in value:# history
            history_response=value
            his={
                #"role": "assistant",
                "role":role,
                "content": [{"type": "text",
                             "text":  history_response}]
            }
            messages.append(his)
        if '<image>' in value:
            png=pngll[img_ix]
            img_ix+=1
            cur_image=os.path.join(inputImgdir,png)
            messages.append(
                {
                #"role": "user",
                    "role":role,
                "content": [{ "type": "image",
                             "image": cur_image,
                             "max_pixels": 28*28*3300,
                            }]}
            )
    return messages
def make_history(historyll,imagelist,drag_prefix=False):
    ### history
    # 顺序 sys+task, history,...,imgPre,history,imgCurrent,->output
    ll = []
    historyll_pre, historyll_last = historyll[:-1], historyll[-1:]
    ### 不包括最后一个的历史
    for h in historyll_pre:
        ll.append(make_history_cell(h))
        # ll.append({
        #     "from": "gpt",
        #     "value": h,  # "history1 thought1",
        #     "loss_mask": 0
        # })
    ##### 图片 imgPre
    if len(imagelist) > 1:
        ll.append(make_image_cell())
        # ll.append({
        #     "from": "human",
        #     "value": "<image> ",
        #     "loss_mask": 0
        # })
    ### 最后一个历史
    for h in historyll_last:
        ll.append(make_history_cell(h))
        # ll.append({
        #     "from": "gpt",
        #     "value": h,  # "history1 thought1",
        #     "loss_mask": 0
        # })
    ### 图片   current img
    ll.append(make_image_cell())
    # ll.append({
    #     "from": "human",
    #     "value": "<image> ",
    #     "loss_mask": 0
    # })
    ### 拖拽终点提示
    if drag_prefix:
        ll.append(make_dragEnd_cell())
        # ll.append({
        #     "from": "human",
        #     "value": drag_prefix,
        #     "loss_mask": 0
        # })
    print('infer shared 186',ll)
    return ll

eos_token_id_map={'uitars':151645,
              'coordfree':151658}
def infer_result(conversation,model,tokenizer,data_processor,assistant_start_placeholder,EOSname):
    return inference(conversation,
              model,
              tokenizer,
              data_processor,
              use_placeholder=assistant_start_placeholder, topk=3,
                    EOS=eos_token_id_map[EOSname])


def get_predict_text(pred):
    return pred['output_text']
def get_predict_pos(pred):
    if pred["topk_points"]:###有交互对象位置
        pos= pred["topk_points"][0]
        px, py=pos
        return px,py
    else:return None### 无交互对象位置