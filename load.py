import cv2
import numpy as np

# 1️⃣ ROI(관심 영역) 설정
def region_of_interest(img):
    height, width = img.shape[:2]
    mask = np.zeros_like(img)

    # 차선이 있는 영역 (다각형)
    polygon = np.array([[
        (int(width * 0.01), int(height * 0.75)),  # 좌측 하단
        (int(width * 0.4), int(height * 0.55)),  # 좌측 상단
        (int(width * 0.6), int(height * 0.55)),  # 우측 상단
        (int(width * 0.99), int(height * 0.75))   # 우측 하단
    ]], np.int32)

    cv2.fillPoly(mask, polygon, (255, 255, 255))
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# 2️⃣ Canny Edge Detection
def detect_edges(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges

# 3️⃣ 허프 변환을 통한 차선 검출
def hough_transform(edges, img):
    lines = cv2.HoughLinesP(edges, rho=2, theta=np.pi/180, threshold=100, minLineLength=50, maxLineGap=50)
    line_image = np.zeros_like(img)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

    return line_image

# 4️⃣ 영상에서 차선 검출
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임 전처리
        roi_img = region_of_interest(frame)
        edges = detect_edges(roi_img)
        lane_lines = hough_transform(edges, frame)

        # 차선 검출된 결과를 원본 이미지에 합성
        result = cv2.addWeighted(frame, 0.8, lane_lines, 1, 0)

        # 결과 출력
        cv2.imshow("Lane Detection", result)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q'를 누르면 종료
            break

    cap.release()
    cv2.destroyAllWindows()

# 📌 실행 (이미 다운로드된 영상 파일 사용)
video_path = "video.mp4"  # 다운로드된 영상 파일명
process_video(video_path)
