# -*- coding: UTF-8 -*-
import os,sys
import argparse
import requests
import cv2
import numpy as np
from multiprocessing import Process
from concurrent.futures import ThreadPoolExecutor
import hashlib
import tldextract
import torch
import json
import pytube
import base64
import datetime
from flask import Flask, request, render_template, redirect, make_response, jsonify
from pathlib import Path
from werkzeug.utils import secure_filename
from main_modules import get_prediction,c_264_2_mp4
from flask_ngrok import run_with_ngrok
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
import math,time
import matplotlib.pyplot as plt
from astropy.utils.data import get_pkg_data_filename
from random import randint
import astropy.io.fits as pyfits
from astropy.table import Table
from collections import Counter
from numpy import mean,var,std
from scipy import stats
from PIL import Image
from detector import Detector

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser('YOLOv5 Online Recognition')
parser.add_argument('--ngrok', action='store_true',
                    default=False, help="Run on local or ngrok")
parser.add_argument('--host',  type=str,
                    default='127.0.0.1:5002', help="Local IP")
parser.add_argument('--debug', action='store_true',
                    default=False, help="Run app in debug mode")
ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))
# 创建线程池执行器
executor = ThreadPoolExecutor(10)
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app, resources={r"/api/*": {"origins": "*"}})  # qing qiu gouzi

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1
UPLOAD_FOLDER = './static/assets/uploads'
VIDEO_FOLDER = './static/assets/videos'
DETECTION_FOLDER = './static/assets/detections'
METADATA_FOLDER = './static/metadata'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DETECTION_FOLDER'] = DETECTION_FOLDER
app.config['VIDEO_FOLDER'] = VIDEO_FOLDER
IMAGE_ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
VIDEO_ALLOWED_EXTENSIONS = {'mp4', 'avi', '3gpp', '3gp',"mov","m4v","mkv"}

def allowed_file_image(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in IMAGE_ALLOWED_EXTENSIONS

def allowed_file_video(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in VIDEO_ALLOWED_EXTENSIONS

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def file_type(path):
    filename = path.split('/')[-1]
    if allowed_file_image(filename):
        filetype = 'image'
    elif allowed_file_video(filename):
        filetype = 'video'
    elif path.endswith(".fits") or path.endswith(".FITS"):
        filetype = "fits"
    else:
        filetype = 'invalid'
    return filetype

def download_yt(url):
    """
    Download youtube video by url and save to video folder
    """
    youtube = pytube.YouTube(url)
    video = youtube.streams.get_highest_resolution()
    path = video.download(app.config['VIDEO_FOLDER'])
    return path

def hash_video(video_path):
    """
    Hash a frame in video and use as a filename
    """
    _, ext = os.path.splitext(video_path)
    stream = cv2.VideoCapture(video_path)
    success, ori_frame = stream.read()
    stream.release()
    stream = None
    image_bytes = cv2.imencode('.jpg', ori_frame)[1].tobytes()
    filename = hashlib.sha256(image_bytes).hexdigest() + f'{ext}'
    return filename

def download(url):
    """
    Handle input url from client
    """
    ext = tldextract.extract(url)
    if ext.domain == 'youtube':
        try:
            make_dir(app.config['VIDEO_FOLDER'])
        except:
            pass
        print('Youtube')
        ori_path = download_yt(url)
        filename = hash_video(ori_path)
        path = os.path.join(app.config['VIDEO_FOLDER'], filename)
        try:
            Path(ori_path).rename(path)
        except:
            pass
    else:
        make_dir(app.config['UPLOAD_FOLDER'])
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36 Edg/95.0.1020.53'}
        r = requests.get(url, headers=headers)
        data = r.content
        print('Image Url',url)
        # Get cache name by hashing image
        ori_filename = url.split('/')[-1]
        _, ext = os.path.splitext(ori_filename)
        filename = hashlib.sha256(data).hexdigest() + f'{ext}'
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(path, "wb") as file:
            file.write(r.content)
    return filename, path


def save_upload(file):
    """
    Save uploaded image and video if its format is allowed
    """
    filename = secure_filename(file.filename)
    if allowed_file_image(filename):
        make_dir(app.config['UPLOAD_FOLDER'])
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    elif allowed_file_video(filename):
        try:
            make_dir(app.config['VIDEO_FOLDER'])
        except:
            pass
        path = os.path.join(app.config['VIDEO_FOLDER'], filename)
    file.save(path)
    return path

path = Path(__file__).parent
@app.route('/ai_api', methods=['POST'])
def api_call():
    if request.method == 'POST':
        response = {}
        #默认返回内容
        if not request.json:
            response['code'] = 404
            response["msg"] = 'NULL'
            return jsonify(response)
        else:

            # get the base64 encoded string
            resource_url = request.json['resourceUrl']
            return_dict = {'code': '200', 'msg': '处理成功'}
            if len(request.get_json()) == 0:
                return_dict['code'] = '5004'
                return_dict['msg'] = '请求参数为空'
                return json.dumps(return_dict, ensure_ascii=False)
            #executor.submit(Ai_detector,imei,resource_url)
            filename, filepath = download(resource_url)
            filetype = file_type(filename)
            filename,aiRecord = Ai_detector(filetype,filename,filepath)
            return_dict = {'code': '200', 'msg': '处理成功'}
            return_dict["filename"] = filename
            return_dict["aiRecord"] = aiRecord
            return json.dumps(return_dict,ensure_ascii=False)

flush_list = False
@app.route('/flush', methods=['POST'])
def flush():
    global flush_list
    r = request.json['flush']
    flush_list = r
    print(flush_list)
    return_dict = {'code': '200', 'msg': '处理成功'}
    return json.dumps(return_dict, ensure_ascii=False)


@app.route('/call_image', methods=['POST'])
def call_image():
    r = request
    nparr = np.fromstring(r.data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    filename = hashlib.sha256(img).hexdigest()
    filetype = "array"
    filename,aiRecord = Ai_detector(filetype, filename, img)
    return_dict = {'code': '200', 'msg': '处理成功'}
    return_dict["filename"] = filename
    return_dict["aiRecord"] = aiRecord
    return json.dumps(return_dict,ensure_ascii=False)


def preprocess_fits(file_name):
    # img_name = ''
    # if file_name.endswith(".fits"):
    #     parts = file_name.split("/")
    #     img_name = parts[-1].replace(".fits", '.png')
    image_file = get_pkg_data_filename(file_name)
    image_data = fits.getdata(image_file, ext=0)
    n, bins, patches = plt.hist(image_data.flatten(), 2560)
    bin_dict = {}
    for v,bin in zip(n, bins):
        bin_dict[bin] = v
    sort_bin = sorted(bin_dict.items(), key = lambda x: x[1], reverse=True)
    Sum = sum(n)
    tmp_sum = 0
    key_list = []
    for pair in sort_bin:
        v = pair[1]
        if tmp_sum/Sum < 0.99:
            tmp_sum += v
            key_list.append(pair[0])
        elif v > 10:
            key_list.append(pair[0])

    mean_key = np.mean(key_list)
    std_key = np.std(key_list)
    upper_bound = mean_key + 3*std_key

    final_key = []
    for k in key_list:
        if k >= upper_bound:
            pass
        else:
            final_key.append(k)
    sort_key =  sorted(final_key)
    vmin = math.ceil(sort_key[1])
    # index = -1
    # for i,c in enumerate(n):
    #     if c < 5:
    #         index = i
    #         break
    vmax = math.floor(sort_key[-1])
    image_data[image_data > vmax] = vmax
    image_data[image_data < vmin] = vmin
    img_data = (image_data-vmin)/(vmax - vmin)
    img_data = (255 * img_data).astype(np.uint8)
    kernel = np.array([[0, 1, 0],
                       [1, 3, 1],
                       [0, 1, 0]], dtype=np.uint8)
    # kernel2 = np.array([[1, 0, 1],
    #                    [0, 1, 0],
    #                    [1, 0, 1]], dtype=np.uint8)
    # kernel = np.ones((3,3),dtype=np.uint8)
    dilate = cv2.dilate(img_data, kernel, 1)
    # res_img_path = 'preprocess/{}'.format(img_name)
    # print(res_img_path)
    # cv2.imwrite(res_img_path, dilate)
    return dilate

@app.route('/read_asset', methods=['POST'])
def read_asset():
    global init_track
    file = request.files.get('file')
    if file is None:
        # 表示没有发送文件
        return_dict = {'code': '400', 'msg': '文件上传失败'}
        return json.dumps(return_dict, ensure_ascii=False)
    file_name = file.filename
    filetype = file_type(file_name)
    print(file_name,filetype)

    if filetype == "image":
        today = datetime.date.today().strftime('%Y%m%d')  # 20220723
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], today)
        if not os.path.exists(upload_path):
            os.makedirs(upload_path)
        if not os.path.exists(os.path.join(upload_path, file_name)):
            file.save(os.path.join(upload_path, file_name))
        filepath = os.path.join(upload_path, file_name)
        img = cv2.imread(filepath)
        t1 = time.time()
        box = detector.detect(img)
        bbox =  []
        for b in box:
            bb = [b[0],b[1],b[2],b[3],"sat",b[5].tolist()]
            bbox.append(bb)
        t2 = time.time()
        return_dict = {}
        return_dict["code"] = "200"
        return_dict["msg"] = "处理成功"
        return_dict["filetype"] = "image"
        return_dict["filename"] = file_name
        return_dict["datetime"] = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        return_dict["num"] = len(bbox)
        return_dict["aiRecord"] = bbox
        return_dict["deal_time"] = round((t2 - t1),3)
        return_dict["engine"] = "sat"
        return json.dumps(return_dict, ensure_ascii=False)

    elif filetype == "fits":
        today = datetime.date.today().strftime('%y%m%d')
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], today)
        out_path = os.path.join(app.config['DETECTION_FOLDER'], today)
        if not os.path.exists(upload_path):
            os.makedirs(upload_path)
        if not os.path.exists(os.path.join(upload_path,file_name)):
            fitspath = os.path.join(upload_path,file_name)
            file.save(fitspath)

        fitspath = os.path.join(upload_path, file_name)
        hdulist = pyfits.open(filepath)
        infos = hdulist.info()
        img_data = hdulist[0].data
        vmin = 0
        vmax = np.max(img_data)
        img_data[img_data > vmax] = vmax
        img_data[img_data < vmin] = vmin
        img_data = (img_data - vmin) / (vmax - vmin)
        img_data = (255 * img_data).astype(np.uint8)
        img_data = img_data[::-1, :]
        img_data = Image.fromarray(img_data, 'L')
        img_path = os.path.join(upload_path, file_name.replace(".fits", ".jpg"))
        img_data.save(img_path)
        img = cv2.imread(img_path)
        t1 = time.time()
        box = detector.detect(img)
        bbox =  []
        for b in box:
            bb = [b[0],b[1],b[2],b[3],"sat",b[5].tolist()]
            print(bb)
            bbox.append(bb)
        t2 = time.time()
        return_dict = {}
        return_dict["code"] = "200"
        return_dict["msg"] = "处理成功"
        return_dict["filetype"] = "fits"
        return_dict["filename"] = file_name
        return_dict["datetime"] = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        return_dict["num"] = 1
        return_dict["aiRecord"] = bbox
        return_dict["deal_time"] = 0.461
        return_dict["engine"] = "sat"
        with open(img_path, 'rb') as image_file:
            return_dict['image'] = base64.b64encode(image_file.read()).decode('utf-8')
        return json.dumps(return_dict, ensure_ascii=False)



if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    if not os.path.exists(DETECTION_FOLDER):
        os.makedirs(DETECTION_FOLDER, exist_ok=True)
    if not os.path.exists(VIDEO_FOLDER):
        os.makedirs(VIDEO_FOLDER, exist_ok=True)
    if not os.path.exists(METADATA_FOLDER):
        os.makedirs(METADATA_FOLDER, exist_ok=True)
    args = parser.parse_args()
    if args.ngrok:
        run_with_ngrok(app)
        app.run()
    else:
        hostname = str.split(args.host, ':')
        if len(hostname) == 1:
            port = 5002
        else:
            port = hostname[1]
        host = hostname[0]
        # app.run(host=host, port=port, debug=args.debug, use_reloader=False,ssl_context='adhoc')
        app.run(host=host, port=port, debug=args.debug)

# Run: python app.py --host localhost:8000
