import numpy as np
import cv2
from findCarPlateNumber import ANPR
import timeit		# Check FPS

# 카메라 대신 재생 할 영상 파일
video = '../testVideo/testVideo.mp4'

# ANPR Class 
anpr = ANPR()


# 찾은 자동차번호판 이미지 처리
def checkBoundingBox(row, img):
	width, height = img.shape[1], img.shape[0]

	# CPU 사용 시 'row[0]' 값은 'tensor(0.32146)' 형태
	# MPS(GPU) 사용시 'row[0]' 값은 'tensor(0.32153, device='mps:0')' 형태
	# row[0].item()을 이용하여 값을 추출
	x1, y1, x2, y2 = int(row[0].item()*width), int(row[1].item()*height), int(row[2].item()*width), int(row[3].item()*height)

	# 자동차번호판 이미지만 잘라내기
	plate_crop = img[int(y1):int(y2), int(x1):int(x2)]
	# 자동차번호판 이미지 테두리에 Green 색으로 그리기
	cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

	# ANPR의 clearImage 호출
	# OCR 처리의 성능을 올리기 위한 처리
	cleared = anpr.clearImage(plate_crop)

	# ANPR의 findPlateCharacters 호출
	# OCR(tesseract를 이용하여) 글자 인식
	result_string = anpr.findPlateCharacters(cleared)

	# tesseract로 번호판 인식하기 직전의 이미지를 Overlay로 출력
	ratio = 250 / (x2 - x1)
	cleared = cv2.resize(cleared, (int((x2 - x1) * ratio), int((y2 - y1) * ratio)))
	cleared = cv2.merge((cleared, cleared, cleared))

	alpha_channel = cleared[:, :, 2] / 255
	overlay_colors = cleared[:, :, :3]

	alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

	h, w = cleared.shape[:2]
	background_subsection = img[0:h, 0:w]

	composite = background_subsection * (1 - alpha_mask) + overlay_colors * alpha_mask

	img[10:h+10, 10:w+10] = composite


	# OCR 결과를 자동차번호판 이미지 위/아래 위치에 출력
	if ((y1 + y2) / 2) > (height / 2):
		cv2.putText(img, result_string, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_4)
	else:
		cv2.putText(img, result_string, (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_4)

	return img




# 테스트 영상 재생
cap = cv2.VideoCapture(video)	# if use camera: option 0

while True:
	# 비디오 읽기
	ret, frame = cap.read()

	# Video Source가 없으면 종료하도록...
	if frame is None: break

	# Check FPS
	start_t = timeit.default_timer()

	# ANPR의 findLabelAndCoordinates 호출
	# Label과 각 Label의 Coordinate를 계산
	labels, coordinates = anpr.findLabelsAndCoordinates(frame)

	if len(labels) > 0:
		frame = [checkBoundingBox(coordinates[i], frame) for i in range(len(labels))][0]

	# Check FPS
	terminate_t = timeit.default_timer()
	FPS = int(1./(terminate_t - start_t ))
	cv2.putText(frame, f'{anpr.mps_device}: {FPS} Fps', (frame.shape[1] - 250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_4)

	# Frame 출력
	cv2.imshow("Car License Plate Check System", frame)

	# 'q' 입력 시 종료
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
        
# Camera 종료 및 Window 종료
cap.release()
cv2.destroyAllWindows()