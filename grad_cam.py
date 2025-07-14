import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

class SmoothGradCAMPlusPlus:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.register_hooks()

    def register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, inputs, target_class):
        outputs = self.model(inputs)
        if target_class is None:
            target_class = torch.argmax(outputs, dim=1).item()

        self.model.zero_grad()
        one_hot_output = torch.zeros_like(outputs)
        one_hot_output[0, target_class] = 1
        outputs.backward(gradient=one_hot_output)

        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))  
        grad_cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            grad_cam += w * activations[i]

        grad_cam = np.maximum(grad_cam, 0)

        grad_cam -= np.min(grad_cam)
        if np.max(grad_cam) > 0:
            grad_cam /= np.max(grad_cam)
        grad_cam = cv2.resize(grad_cam, (inputs.size(-1), inputs.size(-2)))

        return grad_cam

    def apply_smoothing(self, image, target_class, num_samples=50, noise_std=0.2):
        smoothed_cam = np.zeros_like(image[0, 0].cpu().numpy())
        for _ in range(num_samples):
            noise = torch.normal(0, noise_std, size=image.size()).to(image.device)
            noisy_image = image + noise
            cam = self.generate_cam(noisy_image, target_class)
            smoothed_cam += cam

        smoothed_cam -= np.min(smoothed_cam)
        smoothed_cam /= np.max(smoothed_cam)
        
        # plt.imshow(smoothed_cam, cmap='jet')
        # plt.axis("off")
        # plt.show()
        
        return smoothed_cam


if __name__ == "__main__":
    model = models.resnet50(pretrained=True)
    model.eval()

    target_layer = model.layer4[-1].conv3
    cam_generator = SmoothGradCAMPlusPlus(model, target_layer)

    image_path = "/home/nandini/tf/static/malignant/malignant (2)-rotated1-sharpened-rotated2.png"
    image = Image.open(image_path).convert("RGB")
    
    preprocess = transforms.Compose([
        transforms.Resize((124,124)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_image = preprocess(image).unsqueeze(0)

    target_class = None
    smoothed_heatmap = cam_generator.apply_smoothing(input_image, target_class)

    heatmap = cv2.applyColorMap(np.uint8(255 * smoothed_heatmap), cv2.COLORMAP_TURBO)

    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    original_image = np.array(image)
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))

    original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)  # Fix for blending
    overlayed_image = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)

    plt.imshow(cv2.cvtColor(overlayed_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
