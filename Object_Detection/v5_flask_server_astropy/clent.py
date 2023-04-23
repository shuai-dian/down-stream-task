## app
import requests
import base64
## 本地读取图片编码进行传递
# file = open('./sat_00000.0001.jpg', 'rb')
import cv2
filename = 'sat_00000.0001.jpg'
img = cv2.imread('sat_00000.0001.jpg')

print(img.shape)
# _, img_encoded = cv2.imencode('.jpg', img)
# # 将img写入data字典，然后请求服务
# content_type = 'image/jpeg'
# headers = {'content-type': content_type}
# data = {'filename':filename,"img":img_encoded.tostring()}
# resp = requests.post("http://172.21.30.130:35002/call_image", data=data, headers=headers)
#
# import datetime
#
# print(resp)