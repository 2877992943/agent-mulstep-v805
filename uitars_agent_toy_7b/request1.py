import requests
import base64


#address1='https://u468127-8875-10811f68.westx.seetacloud.com:8443'
#url=f'{address1}/uitars'
address='https://u468127-b882-86bb84ef.westx.seetacloud.com:8443'
#address='https://u468127-881c-ef97ab18.westx.seetacloud.com:8443'
urla = f'{address}/action'
urlg = f'{address}/grounding'
api_key = '12345678'

headers = {
        'X-API-KEY': api_key,
        'Content-Type': 'application/json'
    }
def send_post_action(data,urla=urla,headers=headers):
    #### action 发送Post 请求
    response = requests.post(urla, json=data, headers=headers)
    # 打印响应
    #print(response.json())
    data = response.json()
    #print(data['action_output'])
    return data
def send_req(messages):
    # 构造JSON数据
    data = {
        'message': messages
    }

    # 发送POST请求
    response = requests.post(url, json=data)

    # 打印响应
    print('request respons',response.json())
    return response.json()['uitars_output'][0]

def send_fake_showui2b(messages):
    return  """<thought_start>我看到左侧导航栏里有个\"文档\"选项，这正是我需要的。要组织文件和文件夹，文档管理功能是关键。让我点击这个选项，开始进行文件的导入和管理。<thought_end><action_start>{"action': 'LCLICK', 'position': [0.03, 0.16], 'value": None}<action_end><|im_end|>"""
def send_fake_req_7b(messages):
    return "Thought: 我看到界面右上角有设置菜单，我先点击它看看里面有什么\nAction: click(start_box='(666,57)')"
if __name__=='__main__':

    img='./tmp1/20250430@101103.png'
    task='隐藏'





   # 读取图片文件并编码为base64
    with open(img, 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    messages = [
        {
            "role": "user",
            "content": [
                #{"type": "image", "image": f"data:image;base64,/9j/{encoded_image}"},
                {"type": "image","image": f"data:image/png;base64,{encoded_image}"},
                {"type": "text", "text": "describe the image in 100 words"},
            ],
        }
    ]
    # 构造JSON数据
    data = {
        'message': messages
    }

    # 发送POST请求
    response = requests.post(url, json=data)

    # 打印响应
    print(response.json())
    data=response.json()

