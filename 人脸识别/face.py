import cv2
import face_recognition
import os
import numpy as np
path = "img/face_recognition"  #图片目录
cap = cv2.VideoCapture(0) #调用摄像头
total_image_name = []
total_face_encoding = []
image_num = 0
for fn in os.listdir(path):  #fn为图片库文件名
    print(path + "/" + fn)
    total_face_encoding.append(
        face_recognition.face_encodings(
            face_recognition.load_image_file(path + "/" + fn))[0])
    fn = fn[:(len(fn) - 5)]  #截取图片名
    total_image_name.append(fn)  #图片名字列表
    image_num = image_num + 1 #统计图片总数
match_arr = np.zeros(image_num) #初始化存放距离计算结果的数组
results_arr = np.zeros(image_num)
while (1):
    ret, frame = cap.read()
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    # 循环遍历每个人脸
    for (top, right, bottom, left), face_encoding in zip(
            face_locations, face_encodings):
        for i, v in enumerate(total_face_encoding):
            match_arr[i] = np.linalg.norm([v] - face_encoding, ord=np.inf) #根据不同参数以不同的方式计算距离
        match = np.argsort(match_arr)[0]
        name = total_image_name[match] #取距离最短的作为结果
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (255, 0, 0),
                      cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5,
                    (255, 255, 255), 1)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()