import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1️⃣ 이미지 로드
image = cv2.imread("lane_sample.jpg")  # 테스트할 도로 이미지
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 2️⃣ ROI 설정 함수
def draw_roi(img):
    height, width = img.shape[:2]
    mask = np.zeros_like(img)

    # ROI 다각형 정의
    polygon = np.array([[
        (int(width * 0.01), int(height * 0.75)),  # 좌측 하단
        (int(width * 0.4), int(height * 0.55)),  # 좌측 상단
        (int(width * 0.6), int(height * 0.55)),  # 우측 상단
        (int(width * 0.99), int(height * 0.75))   # 우측 하단
    ]], np.int32)

    # 다각형 그리기 (시각적 확인)
    cv2.polylines(img, [polygon], isClosed=True, color=(255, 0, 0), thickness=3)
    
    # 마스크 적용
    cv2.fillPoly(mask, polygon, (255, 255, 255))
    masked_image = cv2.bitwise_and(img, mask)

    return masked_image

# 3️⃣ ROI 테스트 실행
roi_image = draw_roi(image)

# 4️⃣ 결과 시각화
plt.figure(figsize=(10, 6))
plt.imshow(roi_image)
plt.title("ROI Region Visualization")
plt.axis("off")
plt.show()
