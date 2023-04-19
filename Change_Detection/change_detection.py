'''
读取两幅图像。
提取特征点（例如，使用SIFT、SURF或ORB等特征提取方法）。
匹配特征点（例如，使用FLANN近邻搜索）。
应用RANSAC算法进行稳健的特征匹配。
利用对极几何约束（epipolar geometry）计算基础矩阵（fundamental matrix）或单应性矩阵（homography matrix）。
通过基础矩阵或单应性矩阵估计变换矩阵，将一幅图像变换到另一幅图像的视角。
根据变换矩阵对齐两幅图像，计算像素级差异。
应用阈值和形态学操作（如膨胀和腐蚀）来过滤噪声和提取变化区域。
输出变化检测结果。
'''

# from three_order_tucker import change_detection_tucker
import cv2
import numpy as np
import os

from skimage.metrics import structural_similarity as ssim


def mean_squared_error(img1, img2):
    return np.mean((img1 - img2) ** 2)


def extract_region(img, box):
    x1, y1, x2, y2 = box
    return img[y1:y2, x1:x2]


def compare_image_regions(region1, region2):
    # 调整图像区域大小
    region2_resized = cv2.resize(region2, (region1.shape[1], region1.shape[0]), interpolation=cv2.INTER_AREA)
    # mse = mean_squared_error(region1, region2_resized)
    ssim_value = ssim(region1, region2_resized, multichannel=True)

    # return mse, ssim_value
    return ssim_value


def non_max_suppression_no_scores(boxes, iou_threshold):
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.arange(len(boxes))

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep

def mask_to_bbox(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    W,H = mask.shape
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 4 and h < 4:
            pass
        else:
            if w >= h:
                h = w*3
                w = h
            else:
                h = h*3
                w = h
            area = (min(W,(x + w//2)) - max(0,(x - w//4)) ) * (min(H,(y + h//2)) -  max(0,(y - h//4)))
            if area >=0 :
                bounding_boxes.append([max(0,(x - w//4)), max(0,(y - h//4)), min(W,(x + w//2)), min(H,(y + h//2))])
    return bounding_boxes


def filter_noise_edges(mask):
    # 进行形态学操作：闭操作，用于填充小孔
    kernel = np.ones((3, 3), np.uint8)
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # 连通组件分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_closed, connectivity=8)

    # 过滤掉小连通区域
    min_area = 100  # 你可以根据需求调整此值
    filtered_mask = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered_mask[labels == i] = 255

    return filtered_mask

def drawl(image1, image2, matkp1, matkp2):
    width = len(image1[0]) + len(image2[0])
    height = max([len(image1), len(image2)])
    # Change this to np.zeros((height,width,3)) for 3 channel image i.e RGB
    new = np.zeros((height, width))
    new[0:len(image1), 0:len(image1[0])] = image1
    new[0:len(image2), len(image1[0]):len(new[0])] = image2
    for i, j in list(zip(matkp1, matkp2)):
        point1 = (int(i.pt[1]), int(i.pt[0]))
        point2 = (int(j.pt[1]), int(len(image1[0]) + j.pt[0]))

        cv2.line(new, (point1[1], point1[0]), (point2[1], point2[0]), (255, 255, 255), 2)

    return new

def feature_matching(img1, img2, feature_extractor='SIFT', matcher='FLANN'):
    # 1. 特征提取
    if feature_extractor == 'SIFT':
        detector = cv2.SIFT_create()
    elif feature_extractor == 'SURF':
        detector = cv2.xfeatures2d.SURF_create()
    elif feature_extractor == 'ORB':
        detector = cv2.ORB_create()

    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    # 2. 特征匹配
    if matcher == 'FLANN':
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # 3. 应用比率测试以保留好的匹配
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

    return kp1, kp2, good_matches

def change_detection(img1, img2):
    # 特征匹配
    kp1, kp2, good_matches = feature_matching(img1, img2)

    # 计算单应性矩阵
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    h, w = img1.shape[:2]
    # corners1 = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    # corners2 = cv2.perspectiveTransform(corners1.reshape(-1, 1, 2), H)
    # corners2 = corners2.reshape(-1, 2)
    # x1, y1 = np.maximum(np.min(corners2, axis=0), 0).astype(int)
    # x2, y2 = np.minimum(np.max(corners2, axis=0), (w - 1, h - 1)).astype(int)
    # 使用单应性矩阵将图像1变换到图像2的视角
    img1_warped = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))
    gray = cv2.cvtColor(img1_warped, cv2.COLOR_BGR2GRAY)
    # 设置阈值将图像二值化
    thresh = 1
    _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    # cv2.imwrite("img/img1_binary.png",binary)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    maxx,maxy = 0,0
    maxand = 0

    minX = 0
    maxX = w
    minY = 0
    maxY = h
    for i in contours[0]:
        if i[0][0] + i[0][1] > maxand:
            maxx = i[0][0]
            maxy = i[0][1]
            maxand = i[0][0] + i[0][1]

        if i[0][0] < w // 2:
            minX = max(minX,i[0][0])
        else:
            maxX = min(maxX,i[0][0])
        if i[0][1] < h // 2:
            minY = max(minY,i[0][1])
        else:
            maxY = min(maxY,i[0][1])


    x, y, w, h = cv2.boundingRect(np.vstack(contours))
    result = img1_warped[y:maxy, x:maxx]
    # cv2.drawContours(img1_warped, contours, -1, (0, 255, 0), 2)

    # output_img = img1_warped[y1:y2 + 1, x1:x2 + 1]
    cv2.imwrite("img/img1_warped.png",result)
    img2 = img2[y:maxy, x:maxx]
    cv2.imwrite("img/img2_croped.png",img2)

    im1_gry = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    im2_gry = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    ######################
    rank = [500, 500, 3]
    change_map_tucker = change_detection_tucker(im1_gry, im2_gry,rank)
    # cv2.imwrite("img/change_map_002_three.png", change_map_tucker)
    # change_map_tucker = filter_noise_edges(change_map_tucker)
    kernel = np.ones((5, 5), np.uint8)
    diff_eroded_three = cv2.erode(change_map_tucker, kernel, iterations=3)
    diff_dilated_three = cv2.dilate(diff_eroded_three, kernel, iterations=3)
    cv2.imwrite("img/change_map_002_three.png", diff_dilated_three)
    ######################

    # 计算像素级差异
    diff = cv2.absdiff(im1_gry, im2_gry)
    # 应用阈值和形态学操作
    _, diff_thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    diff_thresh = filter_noise_edges(diff_thresh)

    kernel = np.ones((5, 5), np.uint8)
    diff_eroded = cv2.erode(diff_thresh, kernel, iterations=3)
    diff_dilated = cv2.dilate(diff_eroded, kernel, iterations=3)

    # 返回变化检测结果
    return diff_dilated,result,img2,diff_dilated_three


def crop_imgs(img,x0,y0,x1,y1):
    h,w = img.shape[:2]
    if y0 < 0:
        y0 =0
    if x0 < 0 :
        x0 = 0
    if x1 > w:
        x1 = w
    if y1 > h:
        y1 = h
    crop_img = img[y0:y1,x0:x1,:]
    return crop_img

img1 = cv2.imread('img/003/org_044af187b8fa8972_1681269862000.jpg')
img2 = cv2.imread('img/003/org_06612caf0c9dea5b_1681270404000.jpg')

change_mask ,img1_croped,img2_croped,diff_dilated_three = change_detection(img1, img2)

# filtered_diff = filter_noise_edges(change_mask)
# cv2.imwrite("img/002_warped_denoise.png",filtered_diff)

bboxes = mask_to_bbox(change_mask)
# bboxes = mask_to_bbox(diff_dilated_three)
bboxes = np.array(bboxes)
nms_result = non_max_suppression_no_scores(bboxes, iou_threshold = 0.45)
# print(nms_result)
for i, bbox in enumerate(nms_result):
    try:
        bbox = bboxes[i]
        crop_1 = crop_imgs(img1_croped, bbox[0], bbox[1], bbox[2], bbox[3])
        crop_2 = crop_imgs(img2_croped, bbox[0], bbox[1], bbox[2], bbox[3])
        # crop_path = os.path.join("crop_images","{}_{}.png".format(frame_index, i))
        ssim_value = compare_image_regions(crop_1, crop_2)
        # print("均方误差 (MSE):", mse)
        print("结构相似性指数 (SSIM):", ssim_value)
        # cv2.imwrite(crop_path,crop)
        # label, prob = obj_class(cls_net, crop)
        # text = "{} {}".format(label, prob)
        # if label != "":
        #     cv2.putText(img, text, (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        if ssim_value >= 0.7:
            pass
        else:
            cv2.rectangle(img1_croped, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 3)
            cv2.rectangle(img2_croped, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 3)
    except Exception as e:
        print("error:{} box:{}".format(e, bbox))
cv2.imwrite("img/002_img1_det_ret.png",img1_croped)
cv2.imwrite("img/002_img2_det_ret.png",img2_croped)

# cv2.imshow('Change Detection', change_mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
