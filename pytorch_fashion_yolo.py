import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# ==========================
# 1. YOLOv5 로드
# ==========================
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
yolo_model.to('cuda' if torch.cuda.is_available() else 'cpu')
yolo_model.eval()

# ==========================
# 2. FashionMNIST 모델 정의
# ==========================
class FashionMNISTModel(torch.nn.Module):
    def __init__(self):
        super(FashionMNISTModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(64 * 7 * 7, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fmnist_model = FashionMNISTModel().to(device)
fmnist_model.load_state_dict(torch.load('fashion_mnist_model.pth', map_location=device))
fmnist_model.eval()

classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# ==========================
# 3. 데이터 로더 (정확도 측정용)
# ==========================
transform_fmnist = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_fmnist)
test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_fmnist)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

def evaluate(model, dataloader):
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return correct / total

train_acc = evaluate(fmnist_model, train_loader)
test_acc = evaluate(fmnist_model, test_loader)
print(f"[Model Accuracy] Train: {train_acc:.4f}, Test: {test_acc:.4f}")

# ==========================
# 4. YOLO 탐지 + 전처리 함수
# ==========================
transform_input = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def pad_and_resize(img, size=28):
    """비율 유지 + 중앙 정렬 + 패딩"""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))

    delta_w = size - new_w
    delta_h = size - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [255, 255, 255]  # 흰색 배경
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded

def detect_and_classify(image_path, topk=3):
    results = yolo_model(image_path)
    detections = results.xyxy[0].cpu().numpy()

    # 사람, 가방 계열 COCO class ID
    allowed_classes = [0, 24, 26]
    detections = [d for d in detections if int(d[5]) in allowed_classes]
    if len(detections) == 0:
        print("탐지된 대상 없음 (사람/가방 계열만 필터링 중)")
        return

    # 가장 큰 박스 선택
    largest_box = max(detections, key=lambda x: (x[2]-x[0]) * (x[3]-x[1]))
    x1, y1, x2, y2, conf, cls = largest_box

    img_cv = cv2.imread(image_path)
    h, w = img_cv.shape[:2]
    x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(w, x2)), int(min(h, y2))

    crop_img = img_cv[y1:y2, x1:x2]
    crop_img = pad_and_resize(crop_img, 28)

    # 흑백 변환
    img_pil = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)).convert("L")
    img_tensor = transform_input(img_pil).unsqueeze(0).to(device, torch.float)

    with torch.no_grad():
        output = fmnist_model(img_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        top_probs, top_idxs = torch.topk(probs, topk)

    print(f"이미지: {image_path}")
    for i in range(topk):
        label = classes[top_idxs[0][i].item()]
        prob = top_probs[0][i].item()
        print(f"{i+1}. {label} - 확률: {prob*100:.2f}%")

# ==========================
# 5. 실행 예시
# ==========================
image_path = r"C:\data\image1.jpg"
detect_and_classify(image_path, topk=3)
