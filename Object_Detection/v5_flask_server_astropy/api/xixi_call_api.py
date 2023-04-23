# -*- coding: UTF-8 -*-
import requests
import json
from flask import Flask, render_template, Response, request, redirect, url_for, send_from_directory, send_file, jsonify
import time
from requests_toolbelt import MultipartEncoder
import random,os

"""
    call device api method
"""
# @app.route("/call_device_api/", methods=['POST', 'GET'])
def call_device_api():
    api_url = "https://xsh.cloud.neusense.cn/api/xianshanhu-ai/ai/deviceAcquisition/"
    params = {
        'funcationType': 1,
        'current': 1,
        'size': 10
    }
    call_device_api = web_api_Method("GET", api_url, params)  # 采用get方法
    print(call_device_api)

    return jsonify(call_device_api)

def web_api_withMethod(method, url, params):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36 Edg/94.0.992.50",
               "Accept": "text/plain"}
    if method == 'GET':
        r = requests.get(url, params=params, headers=headers)
        api_r = r.text
    elif method == 'POST':
        r = requests.post(url, data=params, headers=headers)
        api_r = r.text
    else:
        api_r = 'The method is not access，only support POST AND GET'
    return api_r

def web_api_Method(method, url, params,):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36 Edg/94.0.992.50",
               "Accept": "text/plain",
               }
    if method == 'GET':
        headers["Accept"] = "application/json"
        headers["Content-Type"] = "application/json;charset=UTF-8"
        r = requests.get(url, params=params, headers=headers)
        api_r = r.text
    # _post commit params chinease char
    elif method == 'POST':
        headers["Accept"] = "application/json"
        headers["Content-Type"] = "application/json;charset=UTF-8"
        r = requests.post(url, data=params, headers=headers)
        api_r = r.text
    else:
        api_r = 'The method is not access，only support POST AND GET'
    return api_r

"""
    call token api method
"""
def call_token_api():
    # api_url = "https://xsh.cloud.neusense.cn/api/blade-auth/oauth/token"
    token_url = "http://192.168.1.46:8323/api/blade-auth/oauth/token"
    params = {
        'grant_type': 'third_party',
        'client_id': 'ai',
        'client_secret': 'ai_secret'
    }

    r = web_api_withMethod("POST", token_url, params)  # 采用post方法
    json_result = json.loads(r)
    access_token = json_result['access_token']
    # print(access_token)
    # r = web_api_withMethod("POST", api_url, params)  # 采用post方法
    return access_token

"""
  call aiList interface
"""
def send_image_info(params):
    # api_url = "https://xsh.cloud.neusense.cn/api/xianshanhu-ai/ai/aiList"
    #api_url = "http://192.168.1.248:8323/xixi-ai/ai/aiList"
    api_url = "http://xixi.data.neusense.cn/xixi-ai/ai/aiList"
    params = json.dumps(params,ensure_ascii=False)
    call_aiList_api = web_api_Method("POST", api_url, params.encode("utf-8"))  # 采用post方法
    print("dfasd:"+call_aiList_api)
    # time.sleep(0.5)

"""
    upload detect bird's images
"""
def upload_image(filename,imei,distinguishTime,aiRecord,resourceType = None):
    # filename  = os.path.join(r"D:\PycharmProjects\flask_web\static\assets\detections",filename)
    # url = 'http://192.168.1.41:7003//api/oss/endpoint/put-file'
    #url = 'http://192.168.1.248:8323/blade-resource/oss/endpoint/put-file'
    url = 'http://xixi.data.neusense.cn/blade-resource/oss/endpoint/put-file'
    # headers = {
    #     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36 Edg/94.0.992.50',
    #     'Referer': url, "Blade-Auth": "bearer eyJ0eXAiOiJKc29uV2ViVG9rZW4iLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJpc3N1c2VyIiwiYXVkIjoiYXVkaWVuY2UiLCJ0ZW5hbnRfaWQiOiIwMDAwMDAiLCJyb2xlX25hbWUiOiJhZG1pbmlzdHJhdG9yIiwicG9zdF9pZCI6IjExMjM1OTg4MTc3Mzg2NzUyMDEiLCJ1c2VyX2lkIjoiMTEyMzU5ODgyMTczODY3NTIwMSIsInJvbGVfaWQiOiIxMTIzNTk4ODE2NzM4Njc1MjAxIiwidXNlcl9uYW1lIjoiYWRtaW4iLCJuaWNrX25hbWUiOiLnrqHnkIblkZgiLCJ0b2tlbl90eXBlIjoiYWNjZXNzX3Rva2VuIiwiZGVwdF9pZCI6IjExMjM1OTg4MTM3Mzg2NzUyMDEiLCJhY2NvdW50IjoiYWRtaW4iLCJjbGllbnRfaWQiOiJzYWJlciIsImV4cCI6MTYzNzcyNzA2NCwibmJmIjoxNjM3NzIzNDY0fQ.E3fJWkE3JzWhxpWox6cGtu2yvi2x--WDgAcH_g4zoRA"
    # }
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36 Edg/94.0.992.50',
        'Referer': url
    }
    multipart_encoder = MultipartEncoder(
        fields={
            'file': (filename, open(filename, 'rb'), 'application/octet-stream')
        },
        boundary='-----------------------------' + str(random.randint(1e12,1e13- 1))
    )
    headers['Content-Type'] = multipart_encoder.content_type

    responseStr = requests.post(url, data=multipart_encoder, headers=headers)
    detect_result = json.loads(responseStr.text)
    print(detect_result)
    if detect_result['code'] == 200:
        params = {
            'imei': imei,
            'distinguishTime': distinguishTime,
            'distinguishUrl': "{}".format(detect_result['data']['link']),
            'resourceType': resourceType,
            'aiRecord': aiRecord
        }
        print(params)
        send_image_info(params)
