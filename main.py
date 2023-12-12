import cv2
# OpenCv와 같이 이미지 처리를 위한 패키지
# Dlib에서 제공하는 정면 얼굴 탐지 모델을 사용. 이 모델은 속도가 빠르고 사용이 쉽다는 장점
import dlib

# 랜드마크는 얼굴의 눈,코 좌표를 뜻합니다.

# Dlib에서 제공하는 이 모델은 정면 얼굴 탐지에 특화된 모델이에요. 
# 아쉽게도 측면 얼굴을 잡아내는 성능이 낮지만 정면은 빠르게 잘 잡아냅니다!
detector = dlib.get_frontal_face_detector()

# 영상과 이미지를 로드
cap = cv2.VideoCapture('videos/01.mp4')
# 오버레이 하고 싶을때는  투명도가 포함된 채널 포함(cv2.IMREAD_UNCHANGED)
sticker_img = cv2.imread('imgs/sticker01.png', cv2.IMREAD_UNCHANGED)

# 동영상 한 프레임 돌리기
while True:
    # 한 프레임씩 읽기
    ret, img = cap.read()
    # 동영상이 끝나면 False
    if ret == False:
        break
    
    # 얼굴영역의 좌표가 표시
    # 전처리를 이 함수가 스스로 해줌
    # 이미지를 넣어주기만 하면 됨, 이 영역의 좌표가 이 dets에 저장
    dets = detector(img)
    # 몇개의 이미지가 인식되었는지 프린트 하기
    print("number of faces detected:", len(dets))

    # 얼굴 수 만큼 loop
    for det in dets:
        x1 = det.left() - 40
        y1 = det.top() - 50
        x2 = det.right() + 40
        y2 = det.bottom() + 30
        print(det)
        # 얼굴영역을 파란색 사각형으로 표시
        # cv2.rectangle(img, pt1=(x1,y1), pt2=(x2,y2), color=(255,0,0), thickness=2)
        try:
            # 복사
            overlay_img = sticker_img.copy()
            #  이미지 얼굴의 사이즈만큼 리사이즈
            overlay_img = cv2.resize(overlay_img, dsize=(x2 - x1, y2 - y1))

            # 투명도 값
            overlay_alpha = overlay_img[:, :, 3:4] / 255.0
            background_alpha = 1.0 - overlay_alpha
            # 오버레이를 실제로 씌우는 부분 1주차때 해봄, 투명한부분 안투명한부분 합치기
            img[y1:y2, x1:x2] = overlay_alpha * overlay_img[:, :, :3] + background_alpha * img[y1:y2, x1:x2]
        except:
            pass    

    cv2.imshow('result', img)
    # 키 입력을 1ms 동안 기다리는데 q버튼 누르면 종료
    if cv2.waitKey(1) == ord('q'):
        break