import torch
from basicsr.metrics.psnr_ssim import calculate_psnr_pt, calculate_ssim_pt

def test(img_tensor, img_tensor2, crop_border, test_y_channel=False):
    # Ensure both tensors are on the GPU
    img_tensor = img_tensor.cuda()
    img_tensor2 = img_tensor2.cuda()

    # --------------------- PyTorch (GPU) ---------------------
    psnr_pth = calculate_psnr_pt(img_tensor, img_tensor2, crop_border=crop_border, test_y_channel=test_y_channel)
    ssim_pth = calculate_ssim_pt(img_tensor, img_tensor2, crop_border=crop_border, test_y_channel=test_y_channel)
   

    # Tensor batch calculation (example with 2x batch size, or just use the batch as is)
    psnr_pth_batch = calculate_psnr_pt(
        torch.repeat_interleave(img_tensor, 2, dim=0),
        torch.repeat_interleave(img_tensor2, 2, dim=0),
        crop_border=crop_border,
        test_y_channel=test_y_channel)
    ssim_pth_batch = calculate_ssim_pt(
        torch.repeat_interleave(img_tensor, 2, dim=0),
        torch.repeat_interleave(img_tensor2, 2, dim=0),
        crop_border=crop_border,
        test_y_channel=test_y_channel)
    
    avg_psnr_batch = psnr_pth_batch.mean()
    avg_ssim_batch = ssim_pth_batch.mean()
    return avg_psnr_batch,avg_ssim_batch 

# if __name__ == '__main__':
#     batch_size = 8
#     img_tensor = torch.randn(batch_size, 3, 512, 512)  # Simulating a batch of 8 images
#     img_tensor2 = torch.randn(batch_size, 3, 512, 512)  # Simulating another batch of 8 images
#     test(img_tensor, img_tensor2, crop_border=4, test_y_channel=False)
