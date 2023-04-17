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


import cv2
import numpy as np

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

    # 使用单应性矩阵将图像1变换到图像2的视角
    img1_warped = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))

    # 计算像素级差异
    diff = cv2.absdiff(img1_warped, img2)

    # 应用阈值和形态学操作
    _, diff_thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    diff_thresh = filter_noise_edges(diff_thresh)



    kernel = np.ones((3, 3), np.uint8)
    diff_dilated = cv2.dilate(diff_thresh, kernel, iterations=3)
    diff_eroded = cv2.erode(diff_dilated, kernel, iterations=3)

    # 返回变化检测结果
    return diff_eroded

img1 = cv2.imread('img/002/org_60fb71ecd432c5c4_1681269902000.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('img/002/org_73fcb9ca7f1531c1_1681270444000.jpg', cv2.IMREAD_GRAYSCALE)


change_mask = change_detection(img1, img2)


filtered_diff = filter_noise_edges(change_mask)

cv2.imwrite("001.png",change_mask)
# cv2.imshow('Change Detection', change_mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
