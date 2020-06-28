import numpy as np
import cv2
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('E:\OneDrive - mails.cqjtu.edu.cn\AllCode\Python\shape_predictor_68_face_landmarks.dat')

# cv2读取图像
img = cv2.imread("E:\OneDrive - mails.cqjtu.edu.cn\AllCode\Python\Face.jpg")

# 取灰度
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# 人脸数rects
rects = detector(img_gray, 0)
for i in range(len(rects)):
    landmarks = np.matrix([[p.x, p.y] for p in predictor(img,rects[i]).parts()])
    for idx, point in enumerate(landmarks):
        # 68点的坐标
        pos = (point[0, 0], point[0, 1])
        print(idx,pos)
        # 利用cv2.circle给每个特征点画一个圈，共68个
        cv2.circle(img, pos, 1, color=(0, 0, 0))
        # 利用cv2.putText输出1-68
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(idx+1), pos, font, 0.3, (0, 0, 255), 1,cv2.LINE_AA)

cv2.namedWindow("img", 2)
cv2.imshow("img", img)
cv2.waitKey(0)