import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import requests
from io import BytesIO

# ==============================
# 1. 디바이스 설정
# ==============================
# Windows
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# MacOS
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# print(f"Using device: {device}")

# ==============================
# 2. 모델 정의
# ==============================
class FashionMNISTModel(nn.Module):
    def __init__(self):
        super(FashionMNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # 28→14
        x = self.pool(torch.relu(self.conv2(x)))  # 14→7
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ==============================
# 3. 데이터 전처리
# ==============================
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 학습/테스트 데이터 로드
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ==============================
# 4. 모델, 손실함수, 옵티마이저
# ==============================
model = FashionMNISTModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# ==============================
# 5. 정확도 계산 함수
# ==============================
def evaluate_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# ==============================
# 6. 학습 루프
# ==============================
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    train_acc = evaluate_accuracy(model, train_loader, device)
    test_acc = evaluate_accuracy(model, test_loader, device)
    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Loss: {running_loss/len(train_loader):.4f} "
          f"Train Acc: {train_acc*100:.2f}% "
          f"Test Acc: {test_acc*100:.2f}%")

# ==============================
# 7. 모델 저장
# ==============================
torch.save(model.state_dict(), 'fashion_mnist_model.pth')
print("모델 저장 완료!")

# ==============================
# 8. 모델 로드
# ==============================
model = FashionMNISTModel()
model.load_state_dict(torch.load('fashion_mnist_model.pth', map_location=device))
model.to(device)
model.eval()

# ==============================
# 9. 예측 함수
# ==============================
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

import torch.nn.functional as F

def predict_image(image_path, topk=3):
    if image_path.startswith("http"):
        response = requests.get(image_path)
        img = Image.open(BytesIO(response.content))
    else:
        img = Image.open(image_path)
    
    img = img.convert("L")
    img = img.resize((28, 28))
    img = transform(img)
    img = img.unsqueeze(0).to(device)
    img = img.to(torch.float)

    with torch.no_grad():
        output = model(img)                   # raw logits
        probs = F.softmax(output, dim=1)     # 확률값 (batch_size x classes)
        top_probs, top_idxs = torch.topk(probs, topk)  # 상위 topk 확률과 인덱스

    print(f"이미지: {image_path}")
    for i in range(topk):
        label = classes[top_idxs[0][i].item()]
        prob = top_probs[0][i].item()
        print(f"{i+1}. {label} - 확률: {prob * 100:.2f}%")


# ==============================
# 10. 예측 실행
# ==============================
image_path = r"C:\data\image1.jpg" # 로컬 이미지 경로
predict_image(image_path,  topk=3)
