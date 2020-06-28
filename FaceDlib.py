import cv2
import dlib
import numpy as np
detector = dlib.get_frontal_face_detector()  #使用默认的人类识别器模型
predictor = dlib.shape_predictor('E:\OneDrive - mails.cqjtu.edu.cn\AllCode\Python\shape_predictor_68_face_landmarks.dat')


def discern(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(gray, 1)
    for face in dets:
        left = face.left()
        top = face.top()
        right = face.right()
        bottom = face.bottom()
        # cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)       #画框框
        cv2.imshow("image", img)


cap = cv2.VideoCapture(0)
while (1):
    ret, img = cap.read()
    # 取灰度
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 人脸数rects
    rects = detector(img_gray, 0)
    for i in range(len(rects)):
        landmarks = np.matrix([[p.x, p.y] for p in predictor(img,rects[i]).parts()])
        for idx, point in enumerate(landmarks):
            # 68点的坐标
            pos = (point[0, 0], point[0, 1])
            # 利用cv2.circle给每个特征点画一个圈，共68个
            cv2.circle(img, pos, 1, color=(0, 0, 0))
            # 利用cv2.putText输出1-68
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, str(idx+1), pos, font, 0.3, (0, 0, 255), 1,cv2.LINE_AA)
        discern(img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()