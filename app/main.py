
import os
import cv2 as cv
from prediction import makePrediction

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
  
  
vid = cv.VideoCapture(0) 
  
while(True): 

    ret, frame = vid.read() 
    
    gray_frame=cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    
    cascade = cv.CascadeClassifier('model/haarcascade_frontalface_default.xml') 
    
    faces_rect = cascade.detectMultiScale(gray_frame, 1.1, 9) 
        
    
    for (x, y, w, h) in faces_rect: 
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    prediction, confidence=makePrediction(image=frame)

    
    if ret:
        if prediction:
            font = cv.FONT_HERSHEY_SIMPLEX
            org = (50, 50)
            fontScale = 1
            color = (0, 255, 0)
            thickness = 2
            cv.putText(frame, prediction, org=org, fontFace=font, fontScale=fontScale, thickness=thickness, color=color)  
        cv.imshow('frame', frame)

    if cv.waitKey(1) & 0xFF == ord('q'): 
        break
  