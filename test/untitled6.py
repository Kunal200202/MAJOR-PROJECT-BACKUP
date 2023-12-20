from ultralytics import YOLO
import cv2


model = YOLO('yolov8n.pt')

#video
video_path = './test.mp4'

cap = cv2.VideoCapture(video_path)

#real-time
# cap = cv2.VideoCapture(0)


frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter("test_output.mp4", cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width, frame_height))

ret = True
while ret:
    ret, frame = cap.read()
    if ret:        
        results = model.track(frame, persist=True, classes=0)        
        frame_ = results[0].plot()    
        cv2.imshow('Detection Output', frame_)
        out.write(frame_)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break