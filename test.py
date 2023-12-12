import cv2
import dlib

detector = dlib.get_frontal_face_detector()
# 랜드마크 모델 로드하기
predictor = dlib.shape_predictor('models/shape_predictor_5_face_landmarks.dat')

cap = cv2.VideoCapture('videos/test/02.mp4')
# 오보레이 이미지를 로드할 경우 cv2.IMREAD_UNCHANGED 사용
sticker_img = cv2.imread('imgs/test/pig.png', cv2.IMREAD_UNCHANGED)

while True:
    ret, img = cap.read()

    if ret == False:
        break

    dets = detector(img)

    try:
        for det in dets:
            shape = predictor(img, det)
            
            for i, point in enumerate(shape.parts()):
                # 점을 그린다. 점의 x,y 좌표, 빨간색
                cv2.circle(img, center=(point.x, point.y), radius=2, color=(0, 0, 255), thickness=-1)
                # 글씨를 쓴다
                cv2.putText(img, text=str(i), org=(point.x, point.y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255, 255, 255), thickness=2)
                
                # 얼굴 좌표
                x1 = det.left()
                y1 = det.top()
                x2 = det.right()
                y2 = det.bottom()

                cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 0), thickness=2)
                
                # compute glasses coordinates
                # 코의 랜드마크는 4번인데 점이 하나 밖에 없기 때문에 돼지코 이미지의 크기를 계산할 때 얼굴의 크기에 비례하도록 만드시는게 좋습니다!
                center_x = shape.parts()[4].x
                center_y = shape.parts()[4].y -5

                h, w, c = sticker_img.shape
                # 코의 크기 가로크기
                nose_w = int((x2 - x1) / 4)
                # 코의 크기 세로크기 ,이미지 비율유지
                nose_h = int(h / w * nose_w)
                # 돼지코 좌표
                nose_x1 = int(center_x - nose_w / 2)                
                nose_x2 = nose_x1 + nose_w
                
                nose_y1 = int(center_y - nose_h / 2)
                nose_y2 = nose_y1 + nose_h
                
                # 오버레이 
                overlay_img = sticker_img.copy()                
                overlay_img = cv2.resize(overlay_img, dsize=(nose_w, nose_h))

                overlay_alpha = overlay_img[:, :, 3:4] / 255.0
                background_alpha = 1.0 - overlay_alpha
                
                img[nose_y1:nose_y2, nose_x1:nose_x2] = overlay_alpha * overlay_img[:, :, :3] + background_alpha * img[nose_y1:nose_y2, nose_x1:nose_x2]
    except Exception as e:      
        print(e)
        pass         

    cv2.imshow('result', img)
    if cv2.waitKey(1) == ord('q'):
        break