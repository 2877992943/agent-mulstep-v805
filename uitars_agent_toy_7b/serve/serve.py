from flask import Flask, request, jsonify
import base64
from PIL import Image
import io
import torch
"""
1 grouding
2 action 其中drag如何出现  在推理一次
based on [demo_infer1_1sample_g_a.py]

<|im_start|>assistant<|recipient|>os\n
<|im_end|>
"""
app = Flask(__name__)
import sys
#sys.path.insert(0,'/root/autodl-tmp/train_code/code616/GUI-Actor-main_lora_616/eval')
from infer_shared import (get_data_process,
                                    get_model,
                                   get_conversation_imgbase64,
                                   #system_message_map,
                                   infer_result,
                                   get_predict_text,
                                   get_predict_pos,
                          make_image_cell,
                          #make_history_cell,
                          make_dragEnd_cell,
                          make_history,
                          get_message2_conversation_imgbase64)

from metric_util import (get_w_h,
                       floatpoint2int)
                      # find_actionType,
                       #find_actionValue,
                      #category1,category2,category3,category4,category5,category6,
                       #drag_end_prompt_addition


date=711
#model_name_or_path = f'/root/autodl-tmp/save{date}/sft_uos_multistep/checkpoint-{step}/merged'
#model_name_or_path='/root/autodl-tmp/save711/qwen2vl_warmup/checkpoint-80000'
#model_name_or_path=f'/root/autodl-tmp/save711/sft_uos_multistep_notplanfirst/checkpoint-{step}/merged'
### autodl-tmp/save711/sft_uos_Iphase8w_data805_newhis_short/checkpoint-34380/merged
### autodl-tmp/save711/sft_uos_Iphase8w_data805_newhis_2img/checkpoint-38220/merged
#step=36507
step=38220
#savename="sft_uos_Iphase8w_data805"
### autodl-tmp/save711/sft_uos_Iphase8w_data805_newhis_2img/checkpoint-38220/merged
savename="sft_uos_Iphase8w_data805_newhis_2img"

model_name_or_path=f'/root/autodl-tmp/save{date}/{savename}/checkpoint-{step}/merged'

data_processor = get_data_process(model_name_or_path)
tokenizer = data_processor.tokenizer
model = get_model(model_name_or_path)
EOS_name=['uitars','coordfree'][0]


API_KEY = '12345678'

@app.before_request
def before_request():
    api_key = request.headers.get('X-API-KEY')
    if api_key != API_KEY:
        return jsonify({'error': 'Unauthorized'}), 401

@app.route('/grounding',methods=['POST'])
@torch.inference_mode()
def grounding():### 没有轨迹 和历史 ，只当前图 只1张图
    data = request.get_json()
    if 'image' not in data or 'obj' not in data:
        return jsonify({'error': 'Missing image or task field'}), 400
    ##### 获取图片和任务
    image_base64 = data['image']
    task = data['obj']
    w, h = data['w'], data['h']
    instruction = task
    ##
    assistant_start_placeholder = True  ## grouding的时候  提供 anchor_token 预测后面的
    convs = [make_image_cell()]

    conversation = get_conversation_imgbase64(instruction,
                                              # system_message=system_message_map['grounding'],
                                              convs,
                                              encoded_string_img=image_base64)
    pred = infer_result(conversation, model, tokenizer, data_processor,
                        assistant_start_placeholder,
                        EOSname=EOS_name)  ### pred结果里有 文本序列  有 [px ,py]
    pos_norm = get_predict_pos(pred)
    if pos_norm:
        pos = floatpoint2int(w, h, pos_norm)
    return jsonify({'grounding_output': pos,
                    'info': f'ver({model_name_or_path})'})

@app.route('/action', methods=['POST'])
@torch.inference_mode()
def action(): # sys+instr + hisN...+ his2+ img2 + his1 + img1(current)
    #参考 mk_meta_CoordFree_manual_multistep_manualThought_notplanfirst2uitarFormat_sys2reThought_newhis_2img.py
    # 获取JSON数据
    data = request.get_json()


    # # 检查数据是否包含必要的字段
    if 'image' not in data or 'obj' not in data:
        return jsonify({'error': 'Missing image or task field'}), 400

    # 获取图片和任务
    image_base64_list = data['image']
    if type(image_base64_list)!=list:image_base64_list=[image_base64_list]

    if len(image_base64_list)>1:
        print('img list ',len(image_base64_list))
    task = data['obj']
    w,h=data['w'],data['h']
    historyll=data['history']
    instruction=task
    ### 类似 mk_xxxx.py中
    ### convs
    # convs = [make_history_cell(h) for h in historyll[:-1]]
    # convs += [make_image_cell()]
    convs=make_history(historyll,image_base64_list)

    assistant_start_placeholder = False  ##
    #### 类似dataset.py中
    conversation = get_message2_conversation_imgbase64(instruction,
                                             # system_message=system_message_map['action'],
                                              convs,
                                              encoded_string_img=image_base64_list)
    pred = infer_result(conversation, 
                        model, 
                        tokenizer, 
                        data_processor,
                        assistant_start_placeholder,
                        EOSname=EOS_name)
    ### 从文本中解析 thought action->actiontype ,value
    text = get_predict_text(pred)## 只判断是否有drag ，具体parse去agent地方完成
    # actionType = find_actionType(text)

    #### 按照不同action type 提取 动作属性
    if  'drag_start' in text:
        ### drag start
        pos_norm_start = get_predict_pos(pred)
        ### drag end
        #convs+=[make_dragEnd_cell()]
        convs=make_history(historyll,image_base64_list,drag_prefix=True)
        conversation = get_message2_conversation_imgbase64(instruction,
                                                  convs,
                                                  #system_message=system_message_map['action'],
                                                  encoded_string_img=image_base64_list)
        pred = infer_result(conversation, 
                            model, 
                            tokenizer, 
                            data_processor, 
                            assistant_start_placeholder,
                           EOSname=EOS_name)
        pos_norm_end = get_predict_pos(pred)
        pos_start = floatpoint2int(w, h, pos_norm_start)
        pos_end = floatpoint2int(w, h, pos_norm_end)
        #actiondict = {'actionType': 'drag', 'position_start': pos_start, 'position_end': pos_end}
        retdict={'text':text,'position_start': pos_start, 'position_end': pos_end}
    else:###非拖拽
        pos_norm = get_predict_pos(pred)

        if pos_norm==None:
            retdict={'text':text }
        else:
            pos = floatpoint2int(w, h, pos_norm)
            retdict = {'text': text,'position':pos,'position_norm':pos_norm}


    # 返回JSON响应
    #log1 = getlog()
    return jsonify({'action_output': retdict,
                    'info':f'ver({model_name_or_path})'})

if __name__ == '__main__':
    app.run(debug=True,port=6006,host='0.0.0.0')
     
