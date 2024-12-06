import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models
import torch.nn as nn

# 超参数设置
batch_size = 16
learning_rate = 0.001
num_epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 自定义数据集
class CustomDataset(Dataset):
    def __init__(self, root_dir, label_map, transform=None):
        """
        Args:
            root_dir (str): 图像文件夹的根目录
            label_map (dict): 映射类别到标签，例如 {'Client': 0, 'Imposter': 1}
            transform (callable, optional): 图像转换操作
        """
        self.img_paths = []
        self.labels = []
        self.transform = transform

        for label_name, label in label_map.items():
            class_dir = os.path.join(root_dir, label_name)
            # 递归遍历子文件夹
            for root, _, files in os.walk(class_dir):
                for img_file in files:
                    file_path = os.path.join(root, img_file)
                    if os.path.isfile(file_path):  # 确保路径是文件
                        self.img_paths.append(file_path)
                        self.labels.append(label)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label

# 数据预处理与加载
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 假设图像尺寸是224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 数据路径和类别映射
root_dir = './NUAA'
label_map = {'Client': 0, 'Imposter': 1}

# 实例化数据集
dataset = CustomDataset(root_dir=root_dir, label_map=label_map, transform=transform)

# 按照8:2划分训练集和验证集
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# 使用预训练的ResNet50模型
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(label_map))  # 修改最后一层以适应类别数量
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练过程
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        # 计算损失和准确率
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # 输出每个epoch的训练信息
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    # 验证集评估
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_epoch_loss = val_loss / len(val_loader)
    val_epoch_accuracy = 100 * val_correct / val_total
    val_losses.append(val_epoch_loss)
    val_accuracies.append(val_epoch_accuracy)

    # 输出验证信息
    print(f'Epoch [{epoch + 1}/{num_epochs}], '
          f'Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.2f}%, '
          f'Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_accuracy:.2f}%')

# 保存模型
torch.save(model.state_dict(), 'NUAA_model.pth')

# 绘制准确率和损失曲线
epochs = np.arange(1, num_epochs + 1)

# 绘制训练和验证的损失曲线
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, val_losses, label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 绘制训练和验证的准确率曲线
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label='Train Accuracy')
plt.plot(epochs, val_accuracies, label='Validation Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# 保存图像
plt.savefig('Epoch100.png')

# 结束
print("训练已结束")
