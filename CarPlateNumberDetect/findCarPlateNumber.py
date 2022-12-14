import torch

import cv2
import numpy as np

# [1st Way]
# tesseract 사용하여 OCR
import pytesseract  


# [2nd Way]
# keras_ocr 사용하여 OCR
# tesseract가 느려, keras_ocr 모듈을 사용해 보려고 시도
# keras_ocr 모듈을 바로 실행하면, 항상 Initial 단계에서 Metal init 등 불필요한 동작을 함
# https://github.com/faustomorales/keras-ocr/tree/master/keras_ocr 에서
# 모델 소스를 받아, Init은 한번만 하고, Predict만 반복해서 하도록 수정
# keras_ocr 모듈 바로 사용 시: 1 FPS
# 소스로 사용 시: 3 FPS
# tesseract 모듈 사용하는 경우와 속도가 같음
# from kerasOCR import Recognizer

class ANPR:
    def __init__(self, minWidth=1, minHeight=4, minArea=80, minRatio=0.1, maxRatio=1.0):
        # Contour 계산 후 노이즈 제거를 위한 초기값
        self.minWidth = minWidth
        self.minHeight = minHeight
        self.minArea = minArea
        self.minRatio = minRatio
        self.maxRatio = maxRatio
        
        # Model summary: 157 layers, 7012822 parameters, 0 gradients, 15.8 GFLOPs
        path = '../yolov5/runs/train/exp9/weights/best.pt'        # YoloV5s (14MB, 6.4ms, 37.2mAP)

        # path = '../yolov5/runs/train/exp12/weights/best.pt'        # YoloV5n (4MB, 6.3ms, 28.4mAP)
        # path = '../yolov5/runs/train/exp13/weights/best.pt'        # YoloV5s (14MB, 6.4ms, 37.2mAP)

        # Model 로드
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                    path = path, 
                                    force_reload=False)

        # Apple Macbook에서 모델을 CPU가 아닌 MPS(GPU)로 실행하기 위하여
        self.mps_device = 'CPU'
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not "
                      "built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ "
                      "and/or you do not have an MPS-enabled device on this machine.")

        else:
            self.mps_device = 'MPS'
            self.model.to(torch.device("mps"))

        # [2nd Way]
        # keras_ocr 이용
        # self.recognizer = Recognizer()

                
    # 로드된 모델을 이용하여 Object Detecting 한 결과 (Labels, Coordinates)를 Return 한다
    def findLabelsAndCoordinates(self, img):
        results = self.model(img)
        return results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]  # labels, coordinates
    
    # OCR을 위한 이미지 처리
    def clearImage(self, img):

        # Convert to Gray Image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     

        # 커널 생성(대상이 있는 픽셀을 강조)
        kernel = np.array([[0, -1, 0],
                            [-1, 9, -1],
                            [0, -1, 0]])

        # 커널 적용 
        gray = cv2.filter2D(gray, -1, kernel) 

        # GaussianBlur
        blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)
        
        # AdaptiveThreshold
        thresh = cv2.adaptiveThreshold(
            blurred,
            maxValue = 255.0,
            adaptiveMethod = cv2.ADAPTIVE_THRESH_MEAN_C,
            thresholdType = cv2.THRESH_BINARY_INV,
            blockSize = 21,
            C=9
        )

        # Find Contours And Filtering
        contours, _ = cv2.findContours(
            thresh,
            mode = cv2.RETR_LIST,
            method = cv2.CHAIN_APPROX_SIMPLE
        )
        
        cleared = thresh.copy()

        # Contour 중에서 번호가 아닌 노이즈로 판단되는 이미지 처리
        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            area = w * h
            ratio = w / h

            if area < self.minArea or w < self.minWidth \
            or h < self.minHeight or ratio < self.minRatio or ratio > self.maxRatio:
                cv2.fillPoly(cleared, pts=[contour], color=(0, 0, 0))

        return cleared
    
    # OCR 처리하여 번호를 판별한다. 
    def findPlateCharacters(self, img):

        # [1st Way]
        # Tesseract를 이용하여 OCR 처리 방법
        # –psm(Page segmentation modes) 옵션: 4, 6, 11 옵션이 한글인식 가능
        # 0 - Orientation and script detection (OSD) only.
        # 1 - Automatic page segmentation with OSD.
        # 2 - Automatic page segmentation, but no OSD, or OCR.
        # 3 - Fully automatic page segmentation, but no OSD. (Default)
        # 4 - Assume a single column of text of variable sizes
        # 5 - Assume a single uniform block of vertically aligned text.
        # 6 - Assume a single uniform block of text.
        # 7 - Treat the image as a single text line.
        # 8 - Treat the image as a single word.
        # 9 - Treat the image as a single word in a circle.
        # 10 - Treat the image as a single character.
        # 11 - Sparse text. Find as much text as possible in no particular order.
        # 12 - Sparse text with OSD.
        # 13 - Raw line. Treat the image as a single text line, bypassing hacks that are Tesseract-specific.


        # -oem(OCR Engine modes) 옵션
        # 0 - Legacy engine only
        # 1 - Neural nets LSTM engine only
        # 2 - Legacy + LSTM engines
        # 3 - Default, based on what is available.

        # Macbook에서 속도저하가 크다. (MPS 활성화 후 12FPS 정도의 영상이 3FPS로 떨어진다.)
        chars = pytesseract.image_to_string(img, lang='enm', config='--psm 7 --oem 1')
    
        result_chars = ''
        for c in chars:
            if ord('A') <= ord(c) <= ord('z') or c.isdigit():
                result_chars += c

        return result_chars.upper()


        # [2nd Way]
        # keras_ocr 이용
        # 모듈을 바로 사용하지 않고, Weight 파일을 읽어 와서 Predict만 반복해서 할 수 있도록 수정
        # https://github.com/faustomorales/keras-ocr/tree/master/keras_ocr
        # 그러나, 속도는 tesseract와 동일하게 3FPS 
        # result_string = self.recognizer.recognize(np.stack((img,)*3, axis=-1))

        # return result_string.upper()        