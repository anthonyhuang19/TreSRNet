import torch
import torch.nn as nn

# Hyperparameters for testing
n_disc = 1  # Train discriminator more frequently

# Loss weights
lambda_mse = 1  # Mean Squared Error (reconstruction) loss weight
lambda_perceptual = 0.05  # Reduced perceptual loss weight (adjust for more detail)
lambda_adversarial = 0.2  # Increased adversarial loss weight (for stronger GAN effects)

# Optimizer settings
learning_rate = 3e-4  # Learning rate for Adam (could try 1e-4 for smoother convergence)
batch_size = 8  # Batch size, larger size helps with stable gradients, but adjust for memory constraints
num_epochs = 200  # Reduced number of epochs, use early stopping to prevent overfitting

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
