# -*- coding: UTF-8 -*-
import os
import argparse
import requests
import cv2
import numpy as np
from multiprocessing import Process
from concurrent.futures import ThreadPoolExecutor
import hashlib
import tldextract
import json
import base64
import datetime,time
from flask import Flask, request, render_template, redirect, make_response, jsonify
from pathlib import Path
from werkzeug.utils import secure_filename
from main_modules import get_prediction,c_264_2_mp4
from flask_ngrok import run_with_ngrok
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
# from api.xixi_call_api import upload_image
# from detector_trt import Detector
import astropy.io.fits as pyfits
from PIL import Image
from detector import Detector
# os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser('YOLOv5 Online Recognition')
parser.add_argument('--ngrok', action='store_true',
                    default=False, help="Run on local or ngrok")
parser.add_argument('--host',  type=str,
                    default='127.0.0.1:5002', help="Local IP")
parser.add_argument('--debug', action='store_true',
                    default=False, help="Run app in debug mode")
ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))
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
IMAGE_ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg',"bmp"}
FIT_ALLOWED_EXTENSIONS = {"fits"}
VIDEO_ALLOWED_EXTENSIONS = {'mp4', 'avi', '3gpp', '3gp',"mov","m4v","mkv"}
engine_file_path = "weights/sat.engine"
detector = Detector()

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
    elif path.endswith(".fits"):
        filetype = "fits"
    else:
        filetype = 'invalid'
    return filetype

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
import io

def get_yolo(size,box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[2]) / 2.0 - 1
    y = (box[1] + box[3]) / 2.0 - 1
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    if w >= 1 :
        w = 0.99
    if h >= 1 :
        h = 0.99
    return (x,y,w,h)

path = Path(__file__).parent
@app.route('/add_img', methods=['POST'])
def add_img():
    data = request.get_data()
    data = json.loads(data)
    print(data)
    filename = data["filename"]
    # f_type = data["type"]
    filetype = file_type(filename)
    img_bir = data["image"]
    de_img = base64.b64decode(img_bir)
    img = Image.open(io.BytesIO(de_img))
    size = img.size
    file_path = os.path.join("../sat_dataset/satellite_train/images",filename)
    label_path = os.path.join("../sat_dataset/satellite_train/labels",filename.replace("jpg",".txt"))
    img.save(file_path)
    boxes = data["boxes"]
    lf = open(label_path,"w",encoding = "UTF-8")
    for i in boxes:
        b = boxes[i]
        yolob = get_yolo(size,b)
        s = "0 " +str(yolob[0]) + ' ' + str(yolob[1]) + " " + str(yolob[2]) + " "+str(yolob[3]) + "\n"
        # lf.write(s)
    lf.close()

    return_dict = {'code': '200', 'msg': '处理成功'}
    return json.dumps(return_dict, ensure_ascii=False)

flush_list = False
@app.route('/flush', methods=['POST'])
def flush():
    global flush_list
    r = request.json['flush']
    flush_list = r
    # print(flush_list)
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
    t1 = time.time()
    box = detector.detect(img)
    t2 = time.time()
    return_dict = {'code': '200', 'msg': '处理成功'}
    return_dict["filetype"] = "image"
    return_dict["filename"] = filetype
    return_dict["datetime"] = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    return_dict["num"] = len(box)
    return_dict["aiRecord"] = box
    return_dict["deal_time"] = round((t2 - t1), 3)
    return_dict["engine"] = "sat"
    return_dict["filename"] = filename
    return_dict["img"] = img
    return json.dumps(return_dict,ensure_ascii=False)

@app.route('/read_asset', methods=['POST'])
def read_asset():
    file = request.files.get('file')
    if file is None:
        # 表示没有发送文件
        return_dict = {'code': '400', 'msg': 'file upload field ! '}
        return json.dumps(return_dict, ensure_ascii=False)
    file_name = file.filename
    filetype = file_type(file_name)

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
            print(bb)
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
        today = datetime.date.today().strftime('%Y%m%d')  # 20220723
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], today)
        if not os.path.exists(upload_path):
            os.makedirs(upload_path)
        if not os.path.exists(os.path.join(upload_path, file_name)):
            file.save(os.path.join(upload_path, file_name))
        filepath = os.path.join(upload_path, file_name)
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
        img_path = os.path.join(upload_path, file_name.replace(".fits",".jpg"))
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
        return_dict["deal_time"] = round((t2-t1),3)
        return_dict["engine"] = "sat"
        with open(img_path, 'rb') as image_file:
            return_dict['image'] = base64.b64encode(image_file.read()).decode('utf-8')
        return json.dumps(return_dict, ensure_ascii=False)

if __name__ == '__main__':
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
