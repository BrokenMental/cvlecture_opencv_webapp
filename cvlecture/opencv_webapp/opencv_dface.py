from django.conf import settings
import numpy as np
import cv2


def opencv_dface(path):
    # 이미지 경로 받아
    img = cv2.imread(path, 1)

    # 이미지 경로가 정상인지 확인
    if (type(img) is np.ndarray):
        # 정상적인 경로라면 이미지를 한번 출력
        print(img.shape)

        # 이미지가 클 경우 사이즈를 줄여서 리사이징
        factor = 1
        if img.shape[1] > 640:
            factor = 640.0 / img.shape[1]
        elif img.shape[0] > 480:
            factor = 480.0 / img.shape[0]

        if factor != 1:
            w = img.shape[1] * factor
            h = img.shape[0] * factor
            img = cv2.resize(img, (int(w), int(h)))

        baseUrl = settings.MEDIA_ROOT_URL + settings.MEDIA_URL

        # 얼굴, 눈 학습 결과 xml을 읽어옴
        # (https://github.com/MareArts/cvlecture_opencv_webapp) 해당 사이트로 이동, 얼굴검출 소스 다운 및 학습결과 xml 파일 import
        face_cascade = cv2.CascadeClassifier(baseUrl + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(baseUrl + 'haarcascade_eye.xml')

        # 얼굴 검출 입력 영상이 흑백이어야 하기 때문에 흑백으로 이미지 변환
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # AdaBoost cascade 함수로 얼굴을 검출
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # 위의 결과물을 합쳐서 검출 결과물을 생성 
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # 검출결과물과 기존 이미지파일을 덮어 씀
        cv2.imwrite(path, img)

    # 이미지 경로 잘못되었을경우 메시지 출력
    else:
        print('someting error')
        print(path)