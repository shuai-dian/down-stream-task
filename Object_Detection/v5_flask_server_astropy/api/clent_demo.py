import requests

url = "http://127.0.0.1:8080/cv"#后端api链接
f=open("cv.jpg",'rb')#以二进制打开前端本地文件
files = {'file':f}#将二进制文件封装为这样一个字典，索引为file
r = requests.post(url=url,files=files)#将文件传送至url所指向的api地址并调用该api结果赋给r
with open('s.jpg','wb') as fw:
    fw.write(r.content)#将返回结果保存为s.jpg
print(r)

