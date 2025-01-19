import cv2
import os
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch import nn
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from TriSRNet import *  # Your model classes (make sure they're correctly imported)
from parameter import *  # Hyperparameters and configuration (make sure this is available)
from loss_function import *  # Loss functions (make sure these are implemented)
from utils import *  # Utility functions (make sure they're available)
from matrix import test
from load_dataset import *

def min_max_normalize(image_tensor):
    image_tensor = image_tensor.float()
    min_val = image_tensor.min()
    max_val = image_tensor.max()
    normalized_image = (image_tensor - min_val) / (max_val - min_val)
    return normalized_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train():
    torch.cuda.empty_cache()

    # Loss functions
    adversarial_loss = nn.BCEWithLogitsLoss().to(device)
    perceptual_loss = VGGLoss(device=device)
    MSE_loss = nn.MSELoss().to(device)
    
    # Model initialization
    generator = TriSRNet(3).to(device)
    discriminator = Discriminator().to(device)
    
    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    

    # Load the dataset with bicubic images as well
    train_dataset = PairedCaptionDataset(root_folders='/home/ants/project/basicsr_dataset')
    train_dataloader = torch.utils.data.DataLoader(train_dataset,num_workers=16,batch_size=batch_size,shuffle=False)
    # Initialize log file
    log_file = open("log_basicsr/training_log_bicubic.txt", "a")
    log_file.write("Epoch, Generator Loss, Discriminator Loss\n")

    # Training loop
    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        
        avg_generator_loss = []
        avg_discriminator_loss = []
        PSNR, SSIM, bicubic_psnr, bicubic_ssim = 0, 0, 0, 0

        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            LR= batch["lr_image"].to(device)
            HR = batch["hr_image"].to(device)
            
            fake = min_max_normalize(generator(LR))
            fake = fake.to(device)    
            # Train Discriminator
            for _ in range(n_disc):    
                optimizer_D.zero_grad()
                real_labels = torch.ones(batch_size, 1).to(device)
                fake_labels = torch.zeros(batch_size, 1).to(device)

                real_loss = adversarial_loss(discriminator(HR), real_labels)
                fake_loss = adversarial_loss(discriminator(fake.detach()), fake_labels)
                
                D_loss = (real_loss + fake_loss) / 2
                
                D_loss.backward()
                optimizer_D.step()
            
            # Train Generator
            optimizer_G.zero_grad()
            G_loss =  MSE_loss(fake, HR) #+ lambda_adversarial * adversarial_loss(discriminator(fake), real_labels) + lambda_perceptual * perceptual_loss(fake, HR)
            G_loss.backward()
            optimizer_G.step()

            avg_generator_loss.append(G_loss.item())
            avg_discriminator_loss.append(D_loss.item())
            
            # Calculate PSNR and SSIM for generated image
            avg_psnr_batch,avg_ssim_batch = test(fake,HR,4)
            PSNR += avg_psnr_batch.item()
            SSIM += avg_ssim_batch.item()
            # if epoch <= 0:
            #     print(avg_psnr_batch.item()," ",avg_ssim_batch.item())
            # imshow(LR, HR,fake, epoch=epoch, title="Image çin " + str(epoch), save_dir="result_basicsr_bicubic/")
        
        # Average losses and metrics
        avg_generator = sum(avg_generator_loss) / len(avg_generator_loss)
        avg_discriminator = sum(avg_discriminator_loss) / len(avg_discriminator_loss)
        
        PSNR = PSNR / len(avg_discriminator_loss)
        SSIM = SSIM / len(avg_discriminator_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Avg Generator Loss: {avg_generator:.4f}, Avg Discriminator Loss: {avg_discriminator:.4f}, Avg PSNR: {PSNR:.4f},Avg SSIM: {SSIM:.4f}")
        
        log_file.write(f"{epoch+1}, {avg_generator:.4f}, {avg_discriminator:.4f}\n")

        if epoch % 10 == 0:  # Save model checkpoints
            torch.save(generator.state_dict(), 'model_basicsr/generator_bicubic1.pth')
            torch.save(discriminator.state_dict(), 'model_basicsr/discriminator_bicubic1.pth')

# Run training
if __name__ == "__main__":
    train()
