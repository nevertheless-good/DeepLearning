import tensorflow as tf
import cv2
import numpy as np
import threading
import os

font = cv2.FONT_HERSHEY_SIMPLEX

# for Model input
img_size = 224

# 학습한 Model 불러오기
new_model = tf.keras.models.load_model('models/model.h5')

# 경고음 울리기
def playAlert():
    frequency = 2500
    duration = 1
    os.system('play -n synth {} sin {}'.format(duration, frequency))

# 얼굴 / 눈 객체 탐지
# 정면 얼굴에서만 눈동자를 판단할 것이 아니므로 face_haarcascade는 사용하지 않는다
# face_haarcascade = 'haarcascade_frontalface_default.xml'
# faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + face_haarcascade)

eyes_haarcascade = 'haarcascade_eye.xml'
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + eyes_haarcascade)


# Camera를 이용한 실시간 이미지 분류
cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     raise IOError("Cannot open webcam")

# 연속으로 눈 감은 횟수
counter = 0

while True:
    # 카메라 읽기
    ret, frame = cap.read()
    # 카메라 좌/우 반전
    frame = cv2.flip(frame, 1)

    # 눈 위치 확인
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eyeCascade.detectMultiScale(gray, 1.1, 4)
    eyes_roi = frame
    x, y, w, h = 0, 0,  0, 0
    for x,y,w,h in eyes:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        eyess = eyeCascade.detectMultiScale(roi_gray)
        if len(eyess) == 0:
            print("eyes are not detected")
        else:
            for (ex,ey,ew,eh) in eyess:
                eyes_roi = roi_color[ey:ey+eh, ex:ex+ew]
                
    # 모델 Input에 맞춰 Size 변경 / Normalize            
    final_image = cv2.resize(eyes_roi, (224, 224))
    final_image = np.expand_dims(final_image, axis = 0)
    final_image = final_image/255.0
    
    # 학습된 모델로 Open/Close 판별
    Predictions = new_model.predict(final_image)
    color = (0, 255, 0)
    if (Predictions > 0.5):
        status = "Open Eyes"
    else:
        counter = counter + 1
        status = "Closed Eyes"
        color = (0, 0, 255)
        
        # 연속으로 눈 감은 횟수 판별
        if counter > 10:
            # 경고 문구 출력
            cv2.putText(frame, "Wake Up!!", (400, 150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
    
            # 경고음 재생 시 화면이 안 끊기도록 다른 Thread에서 소리 재생
            threading.Thread(target=playAlert, args=(), daemon=True).start()
            counter = 0
    
    # Open / Close 상태 출력
    x1,y1,w1,h1 = 0,0,200,75
    cv2.rectangle(frame, (x1, x1), (x1+w1, y1+h1), (0,0,0), -1)
    cv2.putText(frame, status, (x1 + int(w1/10), y1 + int(h1/2)), font, 0.7, color, 2)
 
    # Frame 출력
    cv2.imshow("Drowsiness Alert System", frame)

    # 'q' 입력 시 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# Camera 종료 및 Window 종료
cap.release()
cv2.destroyAllWindows()




