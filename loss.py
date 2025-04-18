import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim# Loss functions (L1 and SSIM loss)



def psnr_loss(y_true, y_pred):
    mse = F.mse_loss(y_true, y_pred)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def ssim_loss(y_true, y_pred):
    return 1 - ssim(y_true.cpu().numpy(), y_pred.cpu().numpy(), multichannel=True)