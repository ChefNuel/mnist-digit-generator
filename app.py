import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------
# CVAE model definition
# -------------------------------------
class CVAE(nn.Module):
    def __init__(self, latent_dim=20):
        super().__init__()
        self.fc1 = nn.Linear(784 + 10, 400)
        self.fc21 = nn.Linear(400, latent_dim)
        self.fc22 = nn.Linear(400, latent_dim)
        self.fc3 = nn.Linear(latent_dim + 10, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x, y):
        h1 = F.relu(self.fc1(torch.cat([x, y], dim=1)))
        return self.fc21(h1), self.fc22(h1)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        h3 = F.relu(self.fc3(torch.cat([z, y], dim=1)))
        return torch.sigmoid(self.fc4(h3))

    def generate(self, digit, num_samples=5):
        z = torch.randn(num_samples, 20)
        y = F.one_hot(torch.tensor([digit] * num_samples), 10).float()
        generated = self.decode(z, y)
        return generated.view(-1, 28, 28).detach().numpy()

# -------------------------------------
# Streamlit UI
# -------------------------------------
st.set_page_config(page_title="Digit Image Generator")
st.title("🧠 Handwritten Digit Image Generator")
st.write("Generate 5 diffrent synthetic MNIST-like images using my own trained model.")

# Select digit
digit = st.selectbox("Choose a digit to generate (0–9):", list(range(10)))

# Load model when button clicked
if st.button("Generate Images"):
    # Load model
    model = CVAE()
    model.load_state_dict(torch.load("cvae_mnist.pth", map_location=torch.device('cpu')))
    model.eval()

    # Generate images
    images = model.generate(digit)

    st.subheader(f"Generated images of digit {digit}")
    cols = st.columns(5)
    for i, col in enumerate(cols):
        fig, ax = plt.subplots()
        ax.imshow(images[i], cmap="gray")
        ax.axis("off")
        col.pyplot(fig)
