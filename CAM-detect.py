import torch
import numpy as np
import cv2
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载训练好的模型
model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # 根据你的类别数量设置
model.load_state_dict(torch.load('NUAA_model.pth', map_location=torch.device('cpu')))
model = model.to(device)
model.eval()

# 类别标签映射
label_map = {0: 'Client', 1: 'Imposter'}

# 预处理函数
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# 读取图片并预处理
def process_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image, image_tensor


# Grad-CAM实现
def generate_grad_cam(model, image_tensor, target_layer):
    # 获取目标层的特征图
    features = []
    gradients = []

    def hook_fn(module, input, output):
        features.append(output)

        # 注册梯度钩子
        def hook_grad(grad):
            gradients.append(grad)

        output.register_hook(hook_grad)

    hook = target_layer.register_forward_hook(hook_fn)

    # 计算模型的预测
    output = model(image_tensor)
    pred_class = torch.argmax(output, dim=1)

    # 反向传播
    model.zero_grad()
    output[0, pred_class].backward()

    # 获取特征图的梯度
    gradient = gradients[0].cpu().data.numpy()[0]
    # 获取特征图
    feature_map = features[0].cpu().data.numpy()[0]

    # 对梯度进行全局平均池化
    pooled_gradients = np.mean(gradient, axis=(1, 2))

    # 使用梯度加权求和特征图
    for i in range(feature_map.shape[0]):
        feature_map[i, :, :] *= pooled_gradients[i]

    # 对每个特征图通道进行求和
    heatmap = np.sum(feature_map, axis=0)

    # 归一化热力图
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / np.max(heatmap)

    hook.remove()

    return heatmap, pred_class.item()


# 可视化Grad-CAM
def display_grad_cam(image_path, heatmap, pred_class):
    # 读取原始图片
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 放大热力图并与图片大小一致
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # 将热力图转换为伪彩色
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]  # 使用Jet伪彩色

    # 将热力图转换为uint8类型
    heatmap_colored = np.uint8(heatmap_colored * 255)

    # 将热力图叠加到原始图像上
    overlayed_image = cv2.addWeighted(image, 0.6, heatmap_colored, 0.4, 0)

    # 添加分类结果文本
    label = label_map[pred_class]
    cv2.putText(overlayed_image, f"Prediction: {label}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 显示结果
    plt.imshow(overlayed_image)
    plt.axis('off')  # 关闭坐标轴
    plt.show()


# 输入图片路径
image_path = 'test/test2.jpg'

# 处理输入图像
image, image_tensor = process_image(image_path)

# 选择ResNet50中的倒数第二层作为目标层
target_layer = model.layer4[2].conv2

# 生成Grad-CAM热力图并获取预测类别
heatmap, pred_class = generate_grad_cam(model, image_tensor, target_layer)

# 可视化结果
display_grad_cam(image_path, heatmap, pred_class)
