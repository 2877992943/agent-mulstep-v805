from flask import Flask, request, jsonify
import requests
import base64
#from PIL import Image
import io
#import torch
"""
1 grouding
2 action 其中drag如何出现  在推理一次
based on [demo_infer1_1sample_g_a.py]
"""
app = Flask(__name__)
import sys
API_KEY = '12345678'

####address = 'https://u468127-a486-f280633a.westx.seetacloud.com:8443'
address='https://u468127-881c-ef97ab18.westx.seetacloud.com:8443'
address='https://u468127-b882-86bb84ef.westx.seetacloud.com:8443'
urla = f'{address}/action'
urlg = f'{address}/grounding'

# 构造请求头部，包含API密钥
headers = {
        'X-API-KEY': API_KEY,
        'Content-Type': 'application/json'
    }




@app.before_request
def before_request():
    api_key = request.headers.get('X-API-KEY')
    if api_key != API_KEY:
        return jsonify({'error': 'Unauthorized'}), 401

@app.route('/grounding',methods=['POST'])
def grounding():
    data = request.get_json()

    if 'image' not in data or 'obj' not in data:
        return jsonify({'error': 'Missing image or task field'}), 400

    response = requests.post(urlg, json=data, headers=headers)
    print('interchange',response.json())
    return jsonify(response.json())


@app.route('/action', methods=['POST'])
def action():
    # 获取JSON数据
    data = request.get_json()

    # 检查数据是否包含必要的字段
    if 'image' not in data or 'obj' not in data:
        return jsonify({'error': 'Missing image or task field'}), 400

    response = requests.post(urla, json=data, headers=headers)
    #print('interchange', response.json())
    return jsonify(response.json())



if __name__ == '__main__':
    app.run(debug=True,port=8506,host='0.0.0.0')
     
