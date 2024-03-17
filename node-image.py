import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchdiffeq import odeint

# Load and preprocess the image
def load_image(image_path, image_size):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)

# Define the Neural ODE architecture for image smoothing
class NeuralODE(nn.Module):
    def __init__(self):
        super(NeuralODE, self).__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.activation = nn.ReLU()

    def forward(self, x):
        dxdt = self.conv(x)
        return dxdt

# Smoothing function using Neural ODE
def smooth_image(image, max_time=1.0):
    model = NeuralODE()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    def ode_func(x, t):
        return model(x)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        x0 = image.clone().detach().requires_grad_(True)
        pred_x = odeint(ode_func, x0, torch.tensor([0, max_time]))[-1]
        loss = criterion(pred_x, image)
        loss.backward()
        optimizer.step()

    return pred_x.squeeze(0)

# Example usage
image_path = 'path/to/your/image.jpg'
image_size = (256, 256)
num_epochs = 100

# Load and preprocess the image
image = load_image(image_path, image_size)

# Perform image smoothing using Neural ODE
smoothed_image = smooth_image(image)

# Convert the smoothed tensor back to an image
smoothed_image = transforms.ToPILImage()(smoothed_image)
smoothed_image.show()
