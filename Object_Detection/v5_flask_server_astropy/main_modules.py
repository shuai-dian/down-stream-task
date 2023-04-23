from re import I
# -*- coding: UTF-8 -*-

import cv2
from PIL import Image
import numpy as np
import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from api import get_info_from_db
import random,json
from detector import Detector
# import tracker
from api.xixi_call_api import upload_image
detector = Detector()

CACHE_DIR = '.cache/'
CSV_FOLDER = './static/csv'
METADATA_FOLDER = './static/metadata'
colors = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for i in
                       range(3600)]  # 颜色列表

def save_cache(aiRecord, cache_name, cache_dir=CACHE_DIR):
    aiRecord = json.dumps(aiRecord)
    with open(f'./{cache_dir}/{cache_name}.json', 'w') as f:
        f.write("{}".format(aiRecord))
    f.close()


def check_cache(cache_name):
    return os.path.isfile(
        f'./{CACHE_DIR}/{cache_name}.csv')  # .cache/d7b0bc6c707bf50750c3acefecde4084560e53faf253b015d1c626f4525d4296_s.txt


def load_cache(image_name):
    df = pd.read_csv(f'./{CACHE_DIR}/{image_name}.csv')
    result_dict = {
        'boxes': [],
        'labels': [],
        'scores': []
    }

    ann = [i for i in zip(df.x, df.y, df.w, df.h, df.labels, df.scores)]

    for row in ann:
        x, y, w, h, label, score = row
        x = float(x)
        y = float(y)
        w = float(w)
        h = float(h)
        box = [x, y, w, h]
        label = int(label)
        score = float(score)
        result_dict['boxes'].append(box)
        result_dict['labels'].append(label)
        result_dict['scores'].append(score)
    return result_dict

def append_food_info(bird_bbox):
    bird_name = set()
    for *xyxy, cls_id, conf in bird_bbox:
        bird_name.add(cls_id)
    bird_info = get_info_from_db(bird_name)
    return bird_info


def convert_dict_to_list(result_dict):
    result_list = []
    num_items = len(result_dict['labels'])
    for i in range(num_items):
        item_dict = {}
        for key in result_dict.keys():
            item_dict[key] = result_dict[key][i]
        result_list.append(item_dict)
    return result_list

def c_264_2_mp4(output):
    ff = 'ffmpeg'
    #result = eval(repr(ff).replace('\\', '/'))
    #ff = result.replace('//', '/')
    video_name = output.split('/')[-1]
    # print(os.path.splitext(output))
    video_path = './static/assets/videos'

    # cmd = fftool + " -i " + file + " -vcodec h264 -threads 5 -preset ultrafast " + newP
    res_path = os.path.join(video_path, video_name)
    if os.path.exists(res_path):
        return res_path
    else:
        cmd = ff + ' -i ' + output + ' -vcodec h264 ' + res_path
        os.system(cmd)
        print('success')
    return res_path 
#c_264_2_mp4('./static/assets/uploads/fe47be1d03b19de0a2f656f18d37a23f90d2491e81a72360d4690317023535f2.mp4')

# def get_video_prediction(
#         input_path,  # ./static/assets/uploads\830ef05fa277fc3d4ec13ccdb5976d4f49449c3ff4b09acc8ea419d4d613e3cc.mp4
#         output_path):
#     ori_hashed_key = os.path.splitext(os.path.basename(input_path))[0]  # f65751e630eecbe53a69bd1ea9e66d74a3cf3b9580ea0f7c974d9167e4cfb2a0
#
#     video_detect = cv2.VideoCapture(input_path)
#     detector = Detector()
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编解码器
#     fps = video_detect.get(cv2.CAP_PROP_FPS)  # 帧数
#     width, height = int(video_detect.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_detect.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 宽高
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))  # 写入视频
#     obj_list = []
#     id_list = []
#     zhuhuan_id = []
#     bird_id = []
#     bailu_id = []
#     aiRecord = []
#     while True:
#         success, img = video_detect.read()
#         if success:
#             img = cv2.resize(img, (width, height))
#             # 获取对象
#             list_bboxs = []
#             bboxes = detector.detect(img)
#             output_image_frame = img.copy()
#             if len(bboxes) > 0:
#                 # list_bboxs = tracker.update(bboxes, img)  # 得到x,y,w,h,cls,id
#                 # cla_list = ["{}".format(b[-2]) for b in list_bboxs]
#                 # id_objs_list = ["{}".format(b[-1]) for b in list_bboxs]
#             for (x1,y1,x2,y2,cls_id,pos_id) in list_bboxs:
#                 if cls_id == "":
#                     cls_id = "bird"
#                 color = colors[pos_id]
#                 label = "{} ID:{}".format(classes_dict_zhuhuan[cls_id], pos_id)
#                 tl = round(0.002 * (output_image_frame.shape[0] + output_image_frame.shape[1]) / 2) + 1  # line/font thickness
#                 c1, c2 = (x1, y1), (x2, y2)
#                 cv2.rectangle(output_image_frame, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
#                 if label:
#                     tf = max(tl - 1, 1)
#                     t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
#
#                     font_size = t_size[1]
#                     font = ImageFont.truetype('./Font/simhei.ttf', font_size)
#                     t_size = font.getsize(label)
#
#                     c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
#                     cv2.rectangle(output_image_frame, c1, c2, color, -1, cv2.LINE_AA)  # filled
#                     # cv2.putText(image, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
#                     # return image
#                     img_PIL = Image.fromarray(cv2.cvtColor(output_image_frame, cv2.COLOR_BGR2RGB))
#                     draw = ImageDraw.Draw(img_PIL)
#                     draw.text((c1[0], c2[1] - 2), label, fill=(255, 255, 255), font=font)
#                     output_image_frame = cv2.cvtColor(np.array(img_PIL), cv2.COLOR_RGB2BGR)
#                 # 画出底部中心点
#                 cv2.circle(output_image_frame, (int(x1 + (x2 - x1) // 2), int(y2)), 5,
#                            colors[pos_id],
#                            cv2.FILLED)
#                 # for i, o in enumerate(obj_list):
#                 #     if o['name'] not in id_objs_list:
#                 #         del obj_list[i]
#                 #     if o['name'] == "{}{}".format(cls_id, pos_id):
#                 #         isNew = False
#                 #         o['points'].append([int(x1 + (x2 - x1) // 2), int(y2)])
#                 if pos_id not in id_list:
#                     if cls_id == "zhuhuan":
#                         zhuhuan_id.append(pos_id)
#                     elif cls_id == "bird":
#                         bird_id.append(pos_id)
#                     elif cls_id == "bailu":
#                         bailu_id.append(pos_id)
#                     id_list.append(pos_id)
#             # output_image_frame = draw_image2(x1, y1, x2, y2, output_image_frame, color, label)
#             out.write(output_image_frame)  # 视频写入
#         else:
#             break
#     if len(zhuhuan_id) != 0 :
#         aiRecord.append({"name":"朱鹮","count":len(zhuhuan_id)})
#     elif len(bailu_id) != 0:
#         aiRecord.append({"name": "白鹭", "count": len(bailu_id)})
#     elif len(bird_id) != 0:
#         aiRecord.append({"name": "其他鸟类", "count": len(bird_id)})
#     #aiRecord = [{"name": "朱鹮", "count": len(zhuhuan_id)}, {"name": "白鹭", "count": len(bailu_id)},{"name": "其他鸟类", "count": len(bird_id)}]
#     save_cache(aiRecord,ori_hashed_key)
#     return output_path,aiRecord


def draw_image2(x1, y1, x2, y2, image, color, label):
    tl = round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (x1, y1), (x2, y2)
    cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]

        font_size = t_size[1]
        font = ImageFont.truetype('./Font/simhei.ttf', font_size)
        t_size = font.getsize(label)

        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
        # cv2.putText(image, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        # return image
        img_PIL = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_PIL)
        draw.text((c1[0], c2[1] - 2), label, fill=(255, 255, 255), font=font)
        return cv2.cvtColor(np.array(img_PIL), cv2.COLOR_RGB2BGR)


def get_prediction(ori_hashed_key,ori_img, output_path):
    # get hashed key from image path
    bboxes = detector.detect(ori_img)
    output_image_frame = ori_img
    aiRecord = {}
    count = 0
    aiRecord["name"] = ori_hashed_key
    aiRecord["number"] = len(bboxes)
    data = []
    for *xyxy, cls_id, conf in bboxes:
        color =(0,255,0)
        score = round(conf.tolist(), 3)
        label = "{}: {}".format("sat", score)
        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
        data.append([x1,y1,x2,y2,cls_id,score])

        tl = round(0.002 * (output_image_frame.shape[0] + output_image_frame.shape[1]) / 2) + 1  # line/font thickness
        c1, c2 = (x1, y1), (x2, y2)
        cv2.rectangle(output_image_frame, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            font_size = t_size[1]
            font = ImageFont.truetype('./Font/simhei.ttf', font_size)
            t_size = font.getsize(label)
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(output_image_frame, c1, c2, color, -1, cv2.LINE_AA)  # filled
            # cv2.putText(image, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
            # return image
            img_PIL = Image.fromarray(cv2.cvtColor(output_image_frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_PIL)
            draw.text((c1[0], c2[1] - 2), label, fill=(255, 255, 255), font=font)
            output_image_frame = cv2.cvtColor(np.array(img_PIL), cv2.COLOR_RGB2BGR)
        count += 1
    aiRecord["data"] = data
    cv2.imwrite(output_path, output_image_frame)
    save_cache(aiRecord, ori_hashed_key)
    print(f"Save cache to {ori_hashed_key}")
    return output_path,aiRecord
