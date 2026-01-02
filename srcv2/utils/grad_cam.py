import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, input_tensor, target_category=None):
        self.model.eval()
        
        # Ensure input tensor is float
        input_tensor = input_tensor.float()
        
        print(f"Input tensor shape in GradCAM: {input_tensor.shape}") # Debug print
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_category is None:
            target_category = torch.argmax(output, dim=1)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        one_hot = torch.zeros_like(output).scatter_(1, target_category.unsqueeze(-1), 1)
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients.cpu().data.numpy()
        activations = self.activations.cpu().data.numpy()
        
        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(2, 3))
        
        # Create heatmap
        heatmap = np.zeros(activations.shape[2:], dtype=np.float32)
        for i, w in enumerate(weights[0]):
            heatmap += w * activations[0, i, :, :]
        
        # ReLU for heatmap
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize heatmap
        heatmap /= np.max(heatmap)
        
        return heatmap

def show_cam_on_image(img: np.ndarray, heatmap: np.ndarray):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    superimposed_img = heatmap * 0.4 + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    return superimposed_img
