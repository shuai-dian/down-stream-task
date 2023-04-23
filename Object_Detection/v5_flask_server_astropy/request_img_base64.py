# Generate the parallel requests based on the ThreadPool Executor
from concurrent.futures import ThreadPoolExecutor as PoolExecutor
import sys
import time
import glob
import requests
import threading
import uuid
import base64
import  json
import os
input_folder = r"D:\code\jms_v5\static\assets\uploads\134e877f14fd257a32302b8b07fa832dd8daa44d5b2523ca4fe0184070f95ac7.jpg"
# # # Encode image into base64 string
# data = {}
# data["filename"] = "134e877f14fd257a32302b8b07fa832dd8daa44d5b2523ca4fe0184070f95ac7.jpg"
# with open (input_folder, 'rb') as image_file:
#     data['image'] =  base64.b64encode(image_file.read()).decode('utf-8')
# data["type"] = "jpg"
# data["boxes"] = {"0":[1,1,100,100],"1":[1,1,100,100]}
# print(data)
# headers = {'Content-Type': 'application/json'}
# response = requests.post("http://172.21.30.130:35002/add_img", json= json.dumps(data), headers = headers)
# print(response)



import requests
import os
import base64
import cv2
import json
import numpy as np

## 本地读取图片编码进行传递
# with open('./44011A0017200303.jpg', 'rb') as f:
#         # image_bytes = f.read()
#         image_bytes = base64.b64encode(f.read())
#         image_bytes = image_bytes.decode('ascii')
# img = image_bytes

# response = requests.get("http://10.130.14.58:16024/autoExtract/downloadImg", {"projectId": "1001120020200110"})
# img = response.text



# with open(input_folder, 'rb') as f:
#         # image_bytes = f.read()
#         image_bytes = base64.b64encode(f.read())
#         image_bytes = image_bytes.decode('ascii')
# img = image_bytes
# data = {'filename':input_folder,"img":img}
# resp = requests.post("http://172.21.30.130:35002/add_img", data=data)
#
# print(resp.text)

# with open(input_folder, 'rb') as image_file:
#     data['image'] = base64.b64encode(image_file.read()).decode('utf-8')
# data["filename"] = input_folder
# url = "http://172.21.30.130:35002/add_img"
# headers = {'Content-Type': 'application/json'}
# response = requests.post(url, json=data, headers=headers)
# print(response)

from PIL import Image


import base64
import requests
with open(input_folder, 'rb') as f:
    im_b64 = base64.b64encode(f.read())

# payload = {'filename': "0001.jpg", 'type': 'jpg', 'boxes': [[0, 0, 100, 100],[0, 0, 100, 100]], 'image': im_b64}
url = 'http://172.21.30.130:5002/add_img'
headers = {'Content-Type': 'application/json'}
data = {}

with open(input_folder, 'rb') as image_file:
    data['image'] = base64.b64encode(image_file.read()).decode('utf-8')
data["filename"] = "0001.jpg"
data["type"] = ".jpg"
boxes = {"0":[0, 0, 100, 100],
         "1":[0, 0, 100, 100]}
data["boxes"] = boxes
response = requests.post(url, json=data, headers=headers)
print(response)




