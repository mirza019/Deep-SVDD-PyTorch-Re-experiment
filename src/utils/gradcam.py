import torch
import numpy as np
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        target_layer.register_backward_hook(self.save_gradient)

    def save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def __call__(self, x):
        x.requires_grad = True
        features, _ = self.model(x)
        score = torch.sum(features)
        self.model.zero_grad()
        score.backward(retain_graph=True)
        grads = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((features * grads).sum(dim=1)).detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam
