import json
import os
import cv2
import re

"""
drag point 2 box
box的起点 终点 调整


"""

category1=['move','lclick','rclick','dlclick','hover']
category2=['drag','drag_start','drag_end']
category3=['hotkey','input(nopos)','input（nopos）']
category4=['input']
category5=['wait','done']
category6=['scroll']### 只有一个位置属性
drag_end_prompt_addition='拖拽动作已经预测起点，现在预测终点'

act1=category1+category6+category2#### only位置属性正确  就正确
act2=category3### only value属性正确 就正确
act3=category5## only  actiontype正确就正确
act4=category4### attr+pos都正确才正确
def get_w_h(imgpath):
    image = cv2.imread(imgpath)
    h,w,_=image.shape
    return w,h

def floatpoint2int(w, h, floatpoint):  ### [x y  ]

    b = [floatpoint[0] * w, floatpoint[1] * h]
    b = [int(v) for v in b]
    return b
