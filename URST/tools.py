import math
from PIL import Image
import torch.nn.functional as F
import torch

def unpadding(image, padding):
    b, c, h ,w = image.shape
    image = image[...,padding:h-padding, padding:w-padding]
    return image

def preprocess(image:Image, padding=32, patch_size=1024, transform=None, cuda=True, square=False):
    W, H = image.size
    N = math.ceil(math.sqrt((W * H) / (patch_size ** 2))) # divide each image dimension through this number of patches to get patches of desired size or smaller
    W_ = math.ceil(W / N) * N + 2 * padding # enlarged image size if using this number of patches + padding
    H_ = math.ceil(H / N) * N + 2 * padding
    w = math.ceil(W / N) + 2 * padding # patch size + padding
    h = math.ceil(H / N) + 2 * padding
    if square:
        w = patch_size + 2 * padding
        h = patch_size + 2 * padding
    if transform is not None:
        image = transform(image)
    image = image.unsqueeze(0)
    
    if cuda:
        image = image.cuda()
    p_left = (W_ - W) // 2
    p_right = (W_ - W) - p_left
    p_top = (H_ - H) // 2
    p_bottom = (H_ - H) - p_top
    image = F.pad(image, [p_left, p_right, p_top, p_bottom], mode="reflect") # reflection pad image to get enlarged sizes calculated above

    b, c, _, _ = image.shape
    images = F.unfold(image, kernel_size=(h, w), stride=(h-2*padding, w-2*padding)) # cut image into padded patches (overlap through striding)
    # Output of unfold is [Batch, Channels*h*w, NumPatches]
    B, C_kw_kw, L = images.shape
    # Reshape to [Batch, NumPatches, Channels, h, w]
    images = images.permute(0, 2, 1).contiguous()
    images = images.view(B, L, c, h, w).squeeze(dim=0) # remove batch dim
    return images

def image_process(image):
    image = image.permute(1, 2, 0).mul_(255.0).add_(0.5).clamp_(0, 255)
    image = image.to(torch.uint8).cpu().data.numpy()
    return image