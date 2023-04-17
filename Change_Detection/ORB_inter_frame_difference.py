import cv2

# 设置视频文件路径
video_path = 'video/20230316150539.mp4'
# 创建OpenCV视频捕获对象
cap = cv2.VideoCapture(video_path)
# 初始化背景模型
fgbg = cv2.createBackgroundSubtractorMOG2()
# 初始化ORB特征检测器
orb = cv2.ORB_create()

while True:
    # 读取视频帧
    ret, frame = cap.read()
    if not ret:
        break
    # 应用背景补偿
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(frame)
    # 运用ORB特征检测
    keypoints = orb.detect(frame, None)
    # # 绘制ORB特征点
    frame = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0), flags=0)
    # # 计算帧间差分
    print(fgbg.shape)
    print(frame.shape)
    diff = cv2.absdiff(gray, fgbg)
    #
    # # 应用阈值
    # thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
    #
    # # 查找轮廓
    # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #
    # # 绘制边界框
    # for contour in contours:
    #     (x, y, w, h) = cv2.boundingRect(contour)
    #     if w > 20 and h > 20:
    #         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    #
    # # 显示视频帧
    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    # 按下q键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放视频捕获对象
cap.release()

# 关闭所有窗口
cv2.destroyAllWindows()