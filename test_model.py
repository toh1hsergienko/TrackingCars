from ultralytics import YOLO
import cv2
# 1:bicycle, 2:car, 3:motorcycle, 5:bus, 7:truck
# Загрузка обученной модели
model = YOLO('C:\ML\yolo11\yolo11n.pt')

cap = cv2.VideoCapture('video driver cars.MOV')

while True:
    # получить один кадр
    ret, frame = cap.read() 
    results = model(frame,classes = [2])
    for result in results:
        boxes = result.boxes.xyxy
        classes = result.boxes.cls
        confidences = result.boxes.conf
        for box, cls, conf in zip(boxes,classes,confidences):
            if cls == 2:
                x1,y1,x2,y2 = map(int,box)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                
    cv2.imshow("Yolo11", frame)
    key = cv2.waitKey(1) 
    if key == 27:
        break
    
cap.release()
cv2.destroyAllWindows()
