import cv2
from ultralytics import YOLO

model=YOLO("yolov8n.pt")  ## If you want to create your own model, you can create a train file and enter the weight file of your own model here.
cap=cv2.VideoCapture(0)

while cap.isOpened():  

    success,frame=cap.read()

    if success:
        results=model.track(frame,tracker="bytetrack.yaml",conf=0.5,iou=0.5) ## also you can use botsort tracker model.

        for result in results:
            if len(result) > 0:
                annotated_frame=result.plot()
            else: 
                print("No Detections and No Tracking")

        cv2.imshow("YoloV8",annotated_frame)
        

        if cv2.waitKey(1) & 0xFF==ord("q"):
            break
        

cap.release()
cv2.destroyAllWindows()

