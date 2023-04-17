import cv2
import numpy as np

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



def find_overlap_bounds(img1, img2, H):
    # 获取图像1的四个角点坐标
    h1, w1 = img1.shape
    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)

    # 使用单应性矩阵将图像1的角点变换到图像2的视角
    corners1_transformed = cv2.perspectiveTransform(corners1, H)

    # 获取变换后的角点坐标的最小和最大值
    x_min, y_min = np.int32(corners1_transformed.min(axis=0).ravel() - 0.5)
    x_max, y_max = np.int32(corners1_transformed.max(axis=0).ravel() + 0.5)

    return x_min, y_min, x_max, y_max

def change_detection(img1, img2):
    # 特征匹配
    kp1, kp2, good_matches = feature_matching(img1, img2)

    # 计算单应性矩阵
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # 使用单应性矩阵将图像1变换到图像2的视角
    img1_warped = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))

    # 找到重叠区域的边界
    x_min, y_min, x_max, y_max = find_overlap_bounds(img1, img2, H)

    # 裁剪重叠区域
    overlap_img1 = img1_warped[y_min:y_max, x_min:x_max]
    overlap_img2 = img2[y_min:y_max, x_min:x_max]

    # 计算像素级差异
    diff = cv2.absdiff(overlap_img1, overlap_img2)

    # 应用阈值和形态学操作
    _, diff_thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    diff_dilated = cv2.dilate(diff_thresh, kernel, iterations=3)
    diff_eroded = cv2.erode(diff_dilated, kernel, iterations=3)

    # 返回变化检测结果
    return diff_eroded

# # 读取图像
# img1 = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)
# img2 = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)
#
# # 变化检测
# change_mask = change_detection(img1, img2)
#
# # 显示结果
# cv2.imshow('Change Detection', change_mask)




img1 = cv2.imread('img/002/org_60fb71ecd432c5c4_1681269902000.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('img/002/org_73fcb9ca7f1531c1_1681270444000.jpg', cv2.IMREAD_GRAYSCALE)


change_mask = change_detection(img1, img2)
cv2.imwrite("001.png",change_mask)