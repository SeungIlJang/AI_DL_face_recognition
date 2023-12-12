import cv2
import dlib

# 얼굴영역 탐지모델
detector = dlib.get_frontal_face_detector()
# 랜드마크 모델 로드하기
predictor = dlib.shape_predictor('models/shape_predictor_5_face_landmarks.dat')

cap = cv2.VideoCapture('videos/01.mp4')
# 오버레이 이미지를 로드할 경우 cv2.IMREAD_UNCHANGED 사용
sticker_img = cv2.imread('imgs/glasses.png', cv2.IMREAD_UNCHANGED)

while True:
    ret, img = cap.read()

    if ret == False:
        break

    dets = detector(img)

    for det in dets:
        # 랜드마크에 이미지와 얼굴모양 넣기
        shape = predictor(img, det)
        # 점은 5개
        for i, point in enumerate(shape.parts()):
            # # 점을 그린다. 점의 x,y 좌표, 빨간색 , 반지름:radius
            # cv2.circle(img, center=(point.x, point.y), radius=2, color=(0, 0, 255), thickness=-1)
            # # 글씨를 쓴다
            # cv2.putText(img, text=str(i), org=(point.x, point.y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255, 255, 255), thickness=2)
            
            # compute glasses coordinates
            # 0:오늘쪽 눈꼬리 1:오른쪽 눈 안쪽
            # 2:왼쪽 눈꼬리 3:왼쪽 눈안쪽
            # 4:코끝
            glasses_x1 = shape.parts()[2].x - 20 # 2:왼쪽 눈꼬리
            glasses_x2 = shape.parts()[0].x + 20 # 0:오늘쪽 눈꼬리

            h, w, c = sticker_img.shape
            # 안경의 가로크기
            glasses_w = glasses_x2 - glasses_x1
            # 안경의 세로크기 ,이미지 비율유지
            glasses_h = int(h / w * glasses_w)
            # 안경의 중심
            center_y = (shape.parts()[0].y + shape.parts()[2].y) / 2
            
            glasses_y1 = int(center_y - glasses_h / 2)
            glasses_y2 = glasses_y1 + glasses_h    
            overlay_img = sticker_img.copy()
            
            overlay_img = cv2.resize(overlay_img, dsize=(glasses_w, glasses_h))

            overlay_alpha = overlay_img[:, :, 3:4] / 255.0
            background_alpha = 1.0 - overlay_alpha
            
            # 안경의 왼쪽 좌표 glasses_x1,glasses_y1
            # 안경의 오른쪽 아래 죄표: glasses_x2,glasses_y2
            # 안경의 너비와 높이: glasses_w, glasses_h
            # 안경의 y축 중심 좌표: center_y
            img[glasses_y1:glasses_y2, glasses_x1:glasses_x2] = overlay_alpha * overlay_img[:, :, :3] + background_alpha * img[glasses_y1:glasses_y2, glasses_x1:glasses_x2]
 
        # try:
        #     x1 = det.left()
        #     y1 = det.top()
        #     x2 = det.right()
        #     y2 = det.bottom()

        #     cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 0), thickness=2)
        # except:
        #     pass

    cv2.imshow('result', img)
    if cv2.waitKey(1) == ord('q'):
        break