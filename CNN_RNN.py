import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 로드
image = cv2.imread("lane_sample.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 1️⃣ ROI(관심 영역) 설정 (사각형)
def region_of_interest(img):
    height, width = img.shape[:2]
    mask = np.zeros_like(img)

    # 차선이 있는 영역 (사각형 형태)
    polygon = np.array([[
        (int(width * 0.01), int(height * 0.9)),   # 좌측 하단
        (int(width * 0.4), int(height * 0.4)),  # 좌측 상단
        (int(width * 0.6), int(height * 0.4)),  # 우측 상단
        (int(width), int(height * 0.9))    # 우측 하단
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

# 3️⃣ 허프 변환으로 차선 검출
def hough_transform(edges):
    lines = cv2.HoughLinesP(edges, rho=2, theta=np.pi/180, threshold=100, minLineLength=50, maxLineGap=50)
    line_image = np.zeros_like(image)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

    return line_image

# 📌 실행
roi_img = region_of_interest(image)
edges = detect_edges(roi_img)
lane_lines = hough_transform(edges)

# 결과 시각화
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(roi_img)
axes[0].set_title("ROI")

axes[1].imshow(edges, cmap="gray")
axes[1].set_title("Canny Edge Detection")

axes[2].imshow(cv2.addWeighted(image, 0.8, lane_lines, 1, 0))
axes[2].set_title("Hough Transform")

plt.show()
