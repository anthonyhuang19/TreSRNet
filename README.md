# TreSRNet-Based Super-Resolution

## Overview
The **TreSRNet-based Super-Resolution** project aims to develop a high-performance image super-resolution model using a Transformer-based architecture. This model incorporates a **Triple Attention Mechanism** (Channel Attention, Spatial Attention, and Self-Attention) to enhance image quality, particularly in terms of detail reconstruction and color accuracy. By leveraging the power of Transformers, it captures global dependencies and focuses on local fine-grained details, resulting in high-resolution image generation from low-resolution inputs.

## Architecture

### 1. RRDB (Residual-in-Residual Dense Block)
The core of the TreSRNet architecture is the **Residual-in-Residual Dense Block (RRDB)**. Each RRDB combines **residual connections** and **dense connections**, which help improve the flow of information across the network and mitigate the vanishing gradient problem. These connections enable the model to retain feature information and improve convergence during training.

- **Residual Connections**: Retain previous activations for better gradient flow.
- **Dense Connections**: Pass all intermediate feature maps to later layers for richer feature representation.

![RRDB Architecture](1.png)

### 2. SR Transformer
The **SR Transformer** is designed to integrate advanced attention mechanisms, allowing the model to learn spatial and channel-wise dependencies effectively. By using **multi-head attention**, it captures both local and global contextual information. This capability is crucial for enhancing the image resolution while preserving fine details in the image.

- **Channel Attention**: Focuses on the importance of each feature map.
- **Spatial Attention**: Highlights important regions of the image.
- **Self-Attention**: Captures long-range dependencies across the image.

![SR Transformer Architecture](2.png)

### 3. Overview of the Architecture
The full architecture of TreSRNet combines **RRDB blocks** and **SR Transformer** to create a robust model capable of generating high-resolution images. The model also uses **upsampling layers** and **convolutional layers** for final refinement, ensuring that the output image retains both fine details and smooth transitions.

- **Upsampling Layer**: Ensures the image is scaled to the desired resolution.
- **Final Convolutional Layer**: Refines the image output with **Tanh activation**.

![Overview of Architecture](3.png)

## Features
- **Triple Attention Mechanism**: Combines Channel Attention, Spatial Attention, and Self-Attention to enhance detail reconstruction and color accuracy.
- **High Performance**: The model achieves higher **PSNR** and **SSIM** scores, outperforming traditional super-resolution models.
- **Real-Time Deployment**: Optimized for real-time image enhancement, suitable for applications like photo enhancement, video upscaling, and live-streaming.
- **State-of-the-art Architecture**: Leverages the powerful Transformer model for contextual understanding and pixel-level precision.

## Installation

### Prerequisites
Before you start, ensure you have Python 3.x installed. It is recommended to create a **virtual environment** for this project to manage dependencies.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repository-link.git
   cd TreSRNet
