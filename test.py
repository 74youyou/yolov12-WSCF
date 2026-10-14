from ultralytics import YOLO
import cv2
img = cv2.imread('zidane.jpg.')
model = YOLO('yolov12n.pt')
results = model.predict(img)
results_image=results[0].plot()
cv2.imwrite("result.jpg", results_image)