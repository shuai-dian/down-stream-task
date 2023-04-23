# -*- coding: UTF-8 -*-
import os
import argparse
import requests
import cv2
import math
import numpy as np
from io import BytesIO
from multiprocessing import Process
from concurrent.futures import ThreadPoolExecutor
import hashlib
import matplotlib.pyplot as plt
import tldextract
import json
import base64
import datetime,time
from flask import Flask, request, render_template, redirect, make_response, jsonify,send_file
from pathlib import Path
from zipfile import ZipFile
from werkzeug.utils import secure_filename
from main_modules import get_prediction,c_264_2_mp4
from flask_ngrok import run_with_ngrok
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
# from api.xixi_call_api import upload_image
# from detector_trt import Detector
import astropy.io.fits as pyfits
from astropy.time import Time

from PIL import Image
from detector import Detector

from sort.sort import *
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)
def draw_boxes(img, bbox, identities=None, categories=None, names=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        cat = int(categories[i]) if categories is not None else 0

        id = int(identities[i]) if identities is not None else 0

        color = compute_color_for_labels(id)

        label = f'{names[cat]} | {id}'
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 +
                                 t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img



# os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser('YOLOv5 Online Recognition')
parser.add_argument('--ngrok', action='store_true',
                    default=False, help="Run on local or ngrok")
parser.add_argument('--host',  type=str,
        default='0.0.0.0:35002', help="Local IP")
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

sort_max_age = 5
sort_min_hits = 3
sort_iou_thresh = 0.2
sort_tracker = Sort(max_age=sort_max_age,
                    min_hits=sort_min_hits,
                    iou_threshold=sort_iou_thresh) # {plug into parser}


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

def preprocess_fits(image_data):
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
        elif v >135:
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
    #dilate = cv2.dilate(img_data, kernel, 1)
    return img_data

path = Path(__file__).parent
@app.route('/add_img', methods=['POST'])
def add_img():
    data = request.get_data()
    data = json.loads(data)
    filename = data["filename"]
    filetype = file_type(filename)
    img_bir = data["image"]
    de_img = base64.b64decode(img_bir)
    img = Image.open(io.BytesIO(de_img))
    size = img.size
    data_path = os.path.join("sat_dataset","satellite_train","images")
    lab_path_dir = os.path.join("sat_dataset","satellite_train","labels")
    
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    if not os.path.exists(lab_path_dir):
        os.makedirs(lab_path_dir)

    file_path = os.path.join(data_path,filename)
    label_path = os.path.join(lab_path_dir ,filename.replace("jpg",".txt"))
    
    img.save(file_path)
    boxes = data["boxes"]
    lf = open(label_path,"w",encoding = "UTF-8")
    for i in boxes:
        b = boxes[i]
        yolob = get_yolo(size,b)
        s = "0 " +str(yolob[0]) + ' ' + str(yolob[1]) + " " + str(yolob[2]) + " "+str(yolob[3]) + "\n"
        # lf.write(s)
    lf.close()

    return_dict = {'code': '200', 'msg': 'success'}
    return json.dumps(return_dict, ensure_ascii=False)

flush_list = False
@app.route('/flush', methods=['POST'])
def flush():
    global flush_list
    r = request.json['flush']
    flush_list = r
    # print(flush_list)
    return_dict = {'code': '200', 'msg': 'success'}
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
    return_dict = {'code': '200', 'msg': 'success'}
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

def is_within_circle(x, y, cx, cy, r):
    d = math.sqrt((x-cx)**2 + (y-cy)**2)
    return d <= r

def write_GTD(GTD_filepath,sat_num,start_time,end_time,s_time,t_time):
    GTD_filename = os.path.basename(GTD_filepath)
    fg = open(GTD_filepath, "w")
    fg.write("C "+ GTD_filename + "\n")
    fg.write("C "+ '\n')
    fg.write("C " + start_time + "\n")
    fg.write("C " + end_time + "\n")
    for i in range(5):
        fg.write("C " + "\n")
    #fg.write("  " + "000"+" " +"0000"+" "+sat_num+" "+"0000"+" "+"0"+" "+"0"+" "+"00"+" "+"0"+" "+s_time+" "+t_time+" "+"\n")
    fg.write("C " + "\n")
    fg.write("END")
    fg.close()



def write_GTW(GTW_filepath,sat_num,start_time,end_time,s_time,t_time):
    GTW_filename = os.path.basename(GTW_filepath)
    fg = open(GTW_filepath,"w")
    fg.write("C "+ GTW_filename + "\n")
    fg.write("C "+ '\n')
    fg.write("C " + start_time + "\n")
    fg.write("C " + end_time + "\n")
    for i in range(5):
        fg.write("C " + "\n")
    #fg.write("  " + "000"+" " +"0000"+" "+sat_num+" "+"0000"+" "+"0"+" "+"0"+" "+"00"+" "+"0"+" "+s_time+" "+t_time+" "+"\n")
    fg.write("C " + "\n")
    fg.write("END")
    fg.close()


@app.route("/upload_fits", methods=["POST"])
def upload():
    #fits_file = request.files["file"]
    fits_file = request.files.get('file')
    file_name = fits_file.filename
    today = datetime.date.today().strftime('%Y%m%d')  # 20220723
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], today) # ./static/assets/uploads/20220723
    if not os.path.exists(upload_path):
        os.makedirs(upload_path)

    file_name = file_name[:-4] + str(time.time()) + ".fits"
    filepath = os.path.join(upload_path, file_name) # ./static/assets/uploads/20220723/****.fits

    if not os.path.exists(filepath):
        fits_file.save(os.path.join(upload_path, file_name))
    
    hdulist = pyfits.open(filepath)
    infos = hdulist.info()
    header = hdulist[0].header
    img_data = hdulist[0].data
    img_data = preprocess_fits(img_data)
    
    img_path = os.path.join(upload_path, file_name.replace(".fits",".jpg"))
    if not os.path.exists(img_path):
        cv2.imwrite(img_path,img_data)
    img = cv2.imread(img_path)
    
    img_w,img_h = img.shape[:2]
    center_x,center_y = img_w //2 ,img_h //2
    center_size = 50

    t1 = time.time()
    dets_to_sort = np.empty((0, 6))
    box = detector.detect(img)
    bbox =  []
    distances = []
    for b in box:
        dets_to_sort = np.vstack((dets_to_sort, np.array([b[0], b[1], b[2], b[3], b[5].tolist(), 0])))
    tracked_dets = sort_tracker.update(dets_to_sort)
    
    obs_date = header["DATE-OBS"]
    obs_date, min_second = obs_date.split(".")
    date, time1 = obs_date.split("T")
    year, month, day = date.split("-")
    hour, minute, second = time1.split(":")
    #t = Time(year=year, month=month, day=day, hour=hour, minute=minute, second=int(second))
    #formatted_date = t.strftime('%Y-%m-%d %H:%M:%S.%f')
    formatted_date = year + month + day + hour + minute + second
    end_time = year + month + day + hour + minute + str(int(second) + 5)
    #main_sat = header["NAME"]
    main_sat = "107756"
    GTD_filename = formatted_date +"_"+ main_sat +"_"+ "0000" + ".GTD"
    GTW_filename = formatted_date +"_"+ main_sat +"_"+ "0000" + ".GTW"

    GTD_filepath = os.path.join("track_log",main_sat,os.sep)
    if not os.path.exists(GTD_filepath):
        os.makedirs(GTD_filename)
    GTD_file_cp = os.path.join(GTD_filepath,GTD_filename)
    GTW_file_cp = os.path.join(GTD_filepath,GTW_filename)
    
    s_time = year + month + day
    t_time = hour, minute, second + min_second[:6]
    if not os.path.exists(GTD_file_cp):
        write_GTD(GTD_file_cp,main_sat,formatted_date,end_time,s_time,t_time)
    
    if not os.path.exists(GTW_file_cp):
        write_GTW(GTW_file_cp,main_sat,formatted_date,end_time,s_time,t_time)

    main_track_id = ""
    for i in tracked_dets:
        track_id = str(int(i[8]))
        track_id = track_id.zfill(6)

        box_center_x = (i[0] + (i[2] - i[0]) // 2)
        box_center_y = (i[1] + (i[3] - i[1]) // 2)
        distance = ((box_center_x - center_x) ** 2 + (box_center_y - center_y)**2) ** 0.5
        if distance <= center_size:
            main_track_id = track_id
        bb = [int(i[0]),int(i[1]),int(i[2]),int(i[3]),track_id,distance]
        bbox.append(bb)

    t2 = time.time()
    track_name = "107756"
    GTD_file_name = obs_date
    
    return_dict = {}
    return_dict["code"] = "200"
    return_dict["msg"] = "success"
    return_dict["filetype"] = "fits"
    return_dict["filename"] = file_name

    try:
        return_dict["DATE-OBS"] = header["DATE-OBS"]
    except Exception as e:
        pass

    return_dict["HUMIDITY"] = 97.5
    return_dict["TEMP"] = 21.3
    return_dict["PRESSURE"] = 98.1
    return_dict["A"] = 203.122409999819
    return_dict["E"] = 31.6131159006528
    return_dict["DIST"] = 38480.4

    return_dict["datetime"] = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    return_dict["num"] = len(bbox)
    return_dict["aiRecord"] = bbox
    return_dict["deal_time"] = round((t2-t1),3)
    return_dict["engine_file"] = "sat"
    
    json_msg_path = os.path.join(upload_path, file_name.replace(".fits",".json"))
    if not os.path.exists(json_msg_path):
        with open(json_msg_path,"w") as f:
            json.dump(return_dict,f)
        f.close()

    #with open(img_path, 'rb') as image_file:
    #    return_dict['image'] = base64.b64encode(image_file.read()).decode('utf-8')
    #return json.dumps(return_dict, ensure_ascii=False)
    zip_file_cp = os.path.join(upload_path, file_name.replace(".fits",".zip"))
    #zip_data = BytesIO()

    with ZipFile(zip_file_cp, mode="w") as archive:
        archive.write("track_log/t1.GTD",arcname = GTD_filename)
        archive.write("track_log/t1.GTW",arcname = GTD_filename.replace("GTD","GTW"))
        archive.write(json_msg_path,arcname = file_name.replace(".fits",".json"))
        archive.write(img_path,arcname = file_name.replace(".fits",".jpg"))
    response = make_response(send_file(zip_file_cp,as_attachment=True))
    #response.headers["Content-Disposition"] = "attachment; filename={}".format(filement.encode().decode("latin-1"))
    response.headers["Content-satname"] = '107756'
    return response
    #return send_file(zip_file_cp,mimetype="application/zip",attachment_filename=file_name.replace(".fits",".zip"), as_attachment=True)
        # return send_file('track_log/t1.GTD', attachment_filename='t1.GTD')W

@app.route('/read_asset', methods=['POST'])
def read_asset():
    file = request.files.get('file')
    if file is None:
        return_dict = {'code': '400', 'msg': 'file upload field ! '}
        return json.dumps(return_dict, ensure_ascii=False)
    file_name = file.filename
    filetype = file_type(file_name)
    if filetype == "image":
        today = datetime.date.today().strftime('%Y%m%d')  # 20220723
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], today)
        if not os.path.exists(upload_path):
            os.makedirs(upload_path)
        file_name = file_name[:-4] + str(time.time()) + ".jpg"
        
        filepath = os.path.join(upload_path, file_name)
        if not os.path.exists(os.path.join(upload_path, file_name)):
            file.save(filepath)

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
        return_dict["msg"] = "success"
        return_dict["filetype"] = "image"
        return_dict["filename"] = file_name
        return_dict["datetime"] = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        return_dict["num"] = len(bbox)
        return_dict["aiRecord"] = bbox
        return_dict["deal_time"] = round((t2 - t1),3)
        return_dict["engine"] = "sat"
        return json.dumps(return_dict, ensure_ascii=False)

    elif filetype == "fits":
        return_dict = {}
        today = datetime.date.today().strftime('%Y%m%d')  # 20220723
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], today)
        if not os.path.exists(upload_path):
            os.makedirs(upload_path)
        file_name = file_name[:-4] + str(time.time()) + ".fits"
        if not os.path.exists(os.path.join(upload_path, file_name)):
            file.save(os.path.join(upload_path, file_name))
        
        filepath = os.path.join(upload_path, file_name)
        hdulist = pyfits.open(filepath)
        infos = hdulist.info()
        header = hdulist[0].header    # FITS header info
        image_data = hdulist[0].data   # FITS images data
        hdulist.close()
        return_dict = {}
        #return_dict["NAME"] = header["NAME"]
        return_dict["NAME"] = '107757'
        return_dict["DATE-OBS"] = header["DATE-OBS"]
        return_dict["HUMIDITY"] = 97.5
        return_dict["TEMP"] = 21.3
        return_dict["PRESSURE"] = 98.1
        return_dict["A"] = 203.122409999819
        return_dict["E"] = 31.6131159006528
        return_dict["DIST"] = 38480.4
        
        img_data = preprocess_fits(image_data)
        img_path = os.path.join(upload_path, file_name.replace(".fits",".jpg"))
        
        cv2.imwrite(img_path,img_data)
        img = cv2.imread(img_path)

        img_w,img_h = img.shape[:2]
        center_x,center_y = img_w //2 ,img_h //2
        center_size = 50
        t1 = time.time()
        box = detector.detect(img)
        bbox =  []
        dets_to_sort = np.empty((0, 6))
        # for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
        #    dets_to_sort = np.vstack((dets_to_sort, np.array([x1, y1, x2, y2, conf, detclass])))
        for b in box:
            dets_to_sort = np.vstack((dets_to_sort, np.array([b[0], b[1], b[2], b[3], b[5].tolist(), 0])))
        tracked_dets = sort_tracker.update(dets_to_sort)
        
        main_track_id = ""
        for i in tracked_dets:
            track_id = str(int(i[8]))
            track_id = track_id.zfill(6)

            box_center_x = (i[0] + (i[2] - i[0]) // 2)
            box_center_y = (i[1] + (i[3] - i[1]) // 2)
            distance = ((box_center_x - center_x) ** 2 + (box_center_y - center_y)**2) ** 0.5

            if distance <= center_size:
                main_track_id = track_id

            bb = [int(i[0]),int(i[1]),int(i[2]),int(i[3]),track_id,distance]
            bbox.append(bb)

        t2 = time.time()
        return_dict["code"] = "200"
        return_dict["msg"] = "success"
        return_dict["filetype"] = "fits"
        return_dict["filename"] = file_name
        return_dict["datetime"] = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        return_dict["num"] = len(bbox)
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
