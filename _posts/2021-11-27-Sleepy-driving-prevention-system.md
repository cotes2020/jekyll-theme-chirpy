---
title: Sleepy driving prevention system
author: Yoon Jiwon, Kim Minjeong
date: 2021-11-27
categories: [Exhibition,2021년]
tags: [post,jiwon,minjeong] 
---

------------------------------------------
# Sleepy driving prevention system

### 티쳐블 머신을 이용해 눈깜빡임 감지 시스템 생성
> 눈을 떴을 때
<img src="/assets/img/post/2021-11-27-Sleepy-driving-prevention-system/tm1.png">

> 눈을 감았을 때
<img src="/assets/img/post/2021-11-27-Sleepy-driving-prevention-system/tm2.png">

### openCV를 이용한 눈깜빡임 감지 시스템
```python
import cv2
import os
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from scipy import io

sleep_classifier = []
for file in files:
  if '.png' in files:
    f = cv2.imread(file)
    sleep_classifier.append(f)

# Face detection XML load and trained model loading
sleep_detection = cv2.CascadeClassifier('haarcascade_eye.xml')
sleep_classifier = image.load_model('주소')
STATES = ["Sleepy" ,"Non-sleepy"]

# Video capture using webcam
camera = cv2.VideoCapture(0)

while True:
    # Capture image from camera
    ret, frame = camera.read()
    
    # Convert color to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Face detection in frame
    sleeps = sleep_detection.detectMultiScale(gray,
                                            scaleFactor=1.1,
                                            minNeighbors=5,
                                            minSize=(30,30))
    
    # Create empty image
    canvas = np.zeros((250, 300, 3), dtype="uint8")
    
    # Perform drowsy driving recognition only when face is detected
    if len(sleeps) > 0:
        # For the largest image
        sleep = sorted(sleeps, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = sleep
        # Resize the image to 48x48 for neural network
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        # Blinking predict
        preds = sleep_classifier.predict(roi)[0]
        blinking_probability = np.max(preds)
        label = STATES[preds.argmax()]
        
        # Assign labeling
        cv2.putText(frame, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)
 
        # Label printing
        for (i, (blinking, prob)) in enumerate(zip(STATES, preds)):
            text = "{}: {:.2f}%".format(blinking, prob * 100)    
            w = int(prob * 300)
            cv2.rectangle(canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), (0, 0, 255), -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

    # Open two windows
    ## Display image ("Drowsy driving recognitionf")
    ## Display probabilities of blinking
    cv2.imshow('Drowsy driving recognition', frame)
    cv2.imshow("Probabilities", canvas)
    
    # q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clear program and close windows
camera.release()
cv2.destroyAllWindows()
```
: 데이터를 로드하는 과정은 생략하였다.

데이터 출처 : <a href = "http://mrl.cs.vsb.cz/eyedataset">

<img src="/assets/img/post/2021-11-27-Sleepy-driving-prevention-system/eyes.png">

### 보완할 점
: 코드에 대한 부분적인 수정이 필요함
(인지 부위를 눈으로 설정한다면 좀 더 효과적으로 졸음으로 인한 눈 blinking을 확인할 수 있을 것이라 생각했다. 하지만 이 부분을 코드로 바꾸는 데에 약간의 어려움이 있어, 얼굴 전체를 인식해 eye blinking 등을 구분해주는 기능까지 구현해보았다.)

