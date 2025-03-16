import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import requests
from io import BytesIO

# Apple Silicon에서 MPS(Metal Performance Shaders) 사용 가능 여부 확인
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# FashionMNIST 모델 정의
class FashionMNISTModel(nn.Module):
    def __init__(self):
        super(FashionMNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # 첫 번째 합성곱 -> ReLU -> MaxPool
        x = self.pool(torch.relu(self.conv2(x)))  # 두 번째 합성곱 -> ReLU -> MaxPool
        x = x.view(-1, 64 * 7 * 7)  # Fully Connected Layer 입력을 위해 Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # 출력층 (10개 클래스)
        return x

# 데이터 전처리 및 변환 정의
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # 모든 이미지를 28x28 크기로 변환
    transforms.ToTensor(),  # 텐서 변환
    transforms.Normalize((0.5,), (0.5,))  # 정규화
])

# FashionMNIST 데이터셋 로드
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 모델 및 학습 설정
model = FashionMNISTModel().to(device)  # 모델을 지정된 디바이스로 이동
criterion = nn.CrossEntropyLoss()  # 다중 클래스 분류 손실 함수
optimizer = Adam(model.parameters(), lr=0.001)  # Adam Optimizer 설정

# 모델 학습 수행
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)  # 데이터를 디바이스로 이동
        optimizer.zero_grad()  # 그래디언트 초기화
        outputs = model(images)  # 모델 예측
        loss = criterion(outputs, labels)  # 손실 계산
        loss.backward()  # 역전파 수행
        optimizer.step()  # 가중치 업데이트
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

# 학습된 모델 저장
torch.save(model.state_dict(), 'fashion_mnist_model.pth')
print("모델 저장 완료!")

# 모델 로드 및 평가 모드 설정
model = FashionMNISTModel()
model.load_state_dict(torch.load('fashion_mnist_model.pth', map_location=device))
model.to(device)  # 디바이스로 이동
model.eval()  # 평가 모드 설정

# FashionMNIST의 클래스 라벨 정의
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 이미지 예측 함수 정의
def predict_image(image_path):
    """
    주어진 이미지 파일을 FashionMNIST 모델을 사용하여 예측합니다.
    """
    # URL에서 이미지 로드 또는 로컬 이미지 불러오기
    if image_path.startswith("http"):
        response = requests.get(image_path)
        img = Image.open(BytesIO(response.content))
    else:
        img = Image.open(image_path)
    
    img = img.convert("L")  # 흑백 변환 (1채널)
    img = img.resize((28, 28))  # 크기 조정 (28x28)
    img = transform(img)  # 변환 적용
    img = img.unsqueeze(0).to(device)  # 배치 차원 추가 (1, 1, 28, 28)
    img = img.to(torch.float)  # 데이터 타입 변환
    
    # 모델을 사용하여 예측 수행
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)
    
    predicted_label = classes[predicted.item()]
    print(f"예측된 옷 종류: {predicted_label}")

# 예측 실행 (로컬 이미지 사용)
image_path = '/Users/parksungsu/Desktop/full-stack/image1.jpg'  # 예시 이미지 경로
predict_image(image_path)