# 匯入函式庫
import cv2
# 設定從哪顆鏡頭讀取影像，本範例中為video0
webcam = cv2.VideoCapture(0)
# 讀取影像
return_value, image = webcam.read()
# 儲存名為Me.png的照片
cv2.imwrite("Me.png", image)
# 刪除webcam，養成不佔用資源的好習慣
del(webcam)
