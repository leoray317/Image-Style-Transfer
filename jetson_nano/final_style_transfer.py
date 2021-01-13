# 匯入函式庫
import numpy as np
import cv2
import time

# 選擇攝影機video0
webcam = cv2.VideoCapture(0)

while(True):
    # 從攝影機擷取影像
    return_value, frame = webcam.read()
    # 加載模型
    net = cv2.dnn.readNetFromTorch('feathers.t7')
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV);
    # 讀取圖片
    image = frame
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (w, h), (103.939, 116.779,123.680), swapRB=False, crop=False)
    # 進行計算
    net.setInput(blob)
    out = net.forward()
    out = out.reshape(3, out.shape[2], out.shape[3])
    out[0] += 103.939
    out[1] += 116.779
    out[2] += 123.68
    out /= 255
    frame = out.transpose(1, 2, 0)

    # 顯示影像
    cv2.imshow("frame", frame)
    time.sleep(1)
    # 按下 q 鍵跳出迴圈
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放攝影機
webcam.release()

# 關閉所有 OpenCV 視窗
cv2.destroyAllWindows()
