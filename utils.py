import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from einops import rearrange, repeat, reduce, pack
from PIL import Image
import warnings

#NOTE: pre and post processing for images, used for Gatys and Deep Feature Synthesis methods
def resize_to_max_size(image: Image, max_size: int, multiple_of_8: bool = False, multiple_of_16: bool = False) -> Image:
    """
    Resize longest side of image to max_size while maintaining aspect ratio. multiple_of_{8,16} ensures
    that the resulting image dimensions are divisible by 8 or 16.
    """
    width, height = image.size

    if multiple_of_16:
        max_size -= max_size % 16
    elif multiple_of_8:
        max_size -= max_size % 8

    if width >= height:
        new_height = int(height * max_size / width)
        if multiple_of_16:
            new_height -= new_height % 16
        elif multiple_of_8:
            new_height -= new_height % 8
        new_size = (max_size, new_height)
    else:
        new_width = int(width * max_size / height)
        if multiple_of_16:
            new_width -= new_width % 16
        elif multiple_of_8:
            new_width -= new_width % 8
        new_size = (new_width, max_size)

    return image.resize(new_size)

def preprocess_image(max_image_size: int, multiple_of_16 = False):
#NOTE: from https://github.com/leongatys/PytorchNeuralStyleTransfer
    prep = transforms.Compose([ 
                            # transforms.Resize(image_size),
                            transforms.ToTensor(),
                            transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                            transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], #subtract imagenet mean
                                                    std=[1,1,1]),
                            transforms.Lambda(lambda x: x.mul_(255)),
                            ])
    prep_resize = lambda x: prep(resize_to_max_size(x, max_image_size, multiple_of_16 = multiple_of_16))
    return prep_resize

def preprocess_greyscale_image(max_image_size: int, multiple_of_16 = False):
    prep = transforms.Compose([
        transforms.Lambda(lambda x: resize_to_max_size(x, max_image_size, multiple_of_16 = multiple_of_16)), 
        transforms.ToTensor()
        ])
    return prep

postpa = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1./255)),
                           transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], #add imagenet mean
                                                std=[1,1,1]),
                           transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to RGB
                           ])

postpb = transforms.Compose([transforms.ToPILImage()])

def postp(tensor): # to clip results in the range [0,1]
#NOTE: from https://github.com/leongatys/PytorchNeuralStyleTransfer
    t = postpa(tensor)
    t[t>1] = 1    
    t[t<0] = 0
    img = postpb(t)
    return img

#NOTE: Luminance/Color remapping for luminance-only style transfer
class LuminanceRemapper():
    """
    Used to remap the color of the generated image to match that of the content image. 
    This is used for luminance-only style transfer.
    This is done by mapping the luminance of the style image to fit the second order statistics of the luminance of the content image and transferring the chrominance channels form the content image to the stylized image.
    If a mask is provided, only the masked region is used to compute the luminance statistics.
    """
    def __init__(self, content_image: Image, content_mask: np.ndarray = None):
        # Compute second order statistics of luminance of content image
        # Convert to YCbCr using the built-in method
        self.content_image_ycbcr = content_image.convert('YCbCr')
        self.content_y = np.array(self.content_image_ycbcr).astype(np.float32)[:, :, 0]
        if content_mask is not None:
            self.content_y_mean = self.content_y[content_mask].mean()
            self.content_y_std = self.content_y[content_mask].std()
        else:
            self.content_y_mean = self.content_y.mean()
            self.content_y_std = self.content_y.std()

    def shift(self, style_image: Image):
        style_ycbcr = np.array(style_image.convert('YCbCr')).astype(np.float32)
        # shift luminance of style image to fit second order statistics of luminance of content image
        style_y = style_ycbcr[:, :, 0]
        style_y_shifted = (style_y - style_y.mean()) * self.content_y_std / (style_y.std() + 1e-9) + self.content_y_mean

        # Clip the values to the valid [0, 255] range for image pixels.
        style_y_shifted_clipped = np.clip(style_y_shifted, 0, 255)
        style_ycbcr_shifted, _ = pack([style_y_shifted_clipped, style_ycbcr[:, :, 1:]], "h w *")

        # convert back to rgb space
        # Suppress warning of more workers than recommended
        warnings.filterwarnings("ignore", category=DeprecationWarning, message="'mode' parameter is deprecated*")
        style_image_shifted = Image.fromarray(style_ycbcr_shifted.astype(np.uint8), mode='YCbCr')

        return style_image_shifted
    
    def unshift(self, output_image: Image):
        # For stylized image, use color channels from content image
        # output_image = transforms.ToPILImage()(output)
        output_image_ycbcr = output_image.convert("YCbCr")
        output_ycbcr = np.array(output_image_ycbcr).astype(np.uint8)
        # resize content colors to fit output size
        output_size = (output_image.size[1], output_image.size[0])
        content_image_ycbcr_resized = transforms.Resize(output_size)(self.content_image_ycbcr)
        content_ycbcr_resized = np.array(content_image_ycbcr_resized).astype(np.float32)
        output_with_color_from_content_ycbcr, _ = pack([output_ycbcr[:, :, 0], content_ycbcr_resized[:, :, 1:].astype(np.uint8)], "h w *")
        output_with_color_from_content_ycbcr = Image.fromarray(output_with_color_from_content_ycbcr, mode = "YCbCr")
        output_with_color_from_content = output_with_color_from_content_ycbcr.convert("RGB")

        return output_with_color_from_content

#NOTE: Loss functions for optimization-based NST
class GramMatrix(nn.Module):
#NOTE: from https://github.com/leongatys/PytorchNeuralStyleTransfer
    def forward(self, input):
        b,c,h,w = input.size()
        F = input.view(b, c, h*w)
        G = torch.bmm(F, F.transpose(1,2)) 
        G.div_(h*w) #NOTE: normalized Gram Matrix
        return G

class MaskedGramMatrix(nn.Module):
    """
    Mask feature statistics when computing the Gram matrix.
    """
    def forward(self, input, mask):
        b,c,h,w = input.size()
        b_mask,c_mask,h_mask,w_mask = mask.size()
        assert h == h_mask and w == w_mask, f"image size = ({h},{w}) does not match mask size = ({h_mask}, {w_mask})"
        F = input.view(b, c, h*w)
        # duplicate mask along channels and batch dim 
        renormalization = mask[0, 0].sum()
        duplicated_mask = repeat(mask, "1 1 h w -> b c h w", b=b, c=c)
        flat_mask = rearrange(duplicated_mask, "b c h w -> b c (h w)") 
        masked_features = flat_mask * F
        G = torch.bmm(masked_features, F.transpose(1,2)) #NOTE: mask Gram Matrix
        # G = torch.bmm(masked_features, masked_features.transpose(1,2)) # mask features
        G.div_(renormalization) #NOTE: renormalized Gram Matrix
        return G

class GramMSELoss(nn.Module):
#NOTE: from https://github.com/leongatys/PytorchNeuralStyleTransfer
    def forward(self, input, target):
        out = nn.MSELoss()(GramMatrix()(input), target) #NOTE: the mean already includes a division with number of channels^2
        return(out)

class MaskedGramMSELoss(nn.Module):
    """
    Apply mask when calculating Gram matrix of input.
    """
    def forward(self, input, target, mask):
        out = nn.MSELoss()(MaskedGramMatrix()(input, mask), target) #NOTE: the mean already includes a division with number of channels^2
        return(out)

class MaskedMSELoss(nn.Module):
    """
    Apply mask when calculating MSELoss.
    """
    def forward(self, input, target, mask):
        b,c,h,w = input.size()
        b_mask,c_mask,h_mask,w_mask = mask.size()
        assert h == h_mask and w == w_mask, f"image size = ({h},{w}) does not match mask size = ({h_mask}, {w_mask})"
        # duplicate mask along channels and batch dim 
        duplicated_mask = repeat(mask, "1 1 h w -> b c h w", b=b, c=c)
        renormalization = h*w / duplicated_mask[0,0].sum()
        out = reduce((input - target).pow(2) * duplicated_mask, "b c h w -> 1", "mean") * renormalization
        return (out)

class MeanVarNorm_MaskedMSELoss(nn.Module):
    """
    Mean-Variance normalized version of MaskedMSELoss. 
    """
    def forward(self, input, target, mask):
        b,c,h,w = input.size()
        b_mask,c_mask,h_mask,w_mask = mask.size()
        assert h == h_mask and w == w_mask, f"image size = ({h},{w}) does not match mask size = ({h_mask}, {w_mask})"
        # duplicate mask along channels and batch dim 
        duplicated_mask = repeat(mask, "1 1 h w -> b c h w", b=b, c=c)
        input_mean, input_std = calc_masked_mean_std(input, duplicated_mask)
        target_mean, target_std = calc_masked_mean_std(target, duplicated_mask)
        input = (input - input_mean) / input_std
        target = (target - target_mean) / target_std
        out = nn.MSELoss()(input * duplicated_mask, target * duplicated_mask) * h*w / duplicated_mask[0, 0].sum() 
        return (out)

def calc_mean_std(feat, eps=1e-6):
    """
    Calculates mean and std over height and width of an image (batch and channel wise).
    """
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2)
    feat_std = feat_var.sqrt().view(N, C, 1, 1) + eps
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def calc_masked_mean_std(feat, mask, eps=1e-6):
    """
    Calculates mean and std over height and width of masked region of an image (batch and channel wise).
    """
    size = feat.size()
    assert (len(size) == 4)
    N, C, H, W = size
    masked_feat = feat * mask

    # distribution masking
    renormalization = H*W / mask[0, 0].sum()
    feat_mean = masked_feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1) * renormalization 
    feat_var = (torch.pow((feat - feat_mean), 2) * mask).view(N, C, -1).mean(dim=2) * renormalization

    # feature masking
    # feat_mean = masked_feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1) # feature masking
    # feat_var = (torch.pow((masked_feat - feat_mean), 2)).view(N, C, -1).mean(dim=2) # feature masking

    # feat_var = ((feat - feat_mean) * mask).pow(2).view(N, C, -1).mean(dim=2) * renormalization # squared mask
    feat_std = feat_var.sqrt().view(N, C, 1, 1) + eps
    return feat_mean, feat_std

class Mean_Std_MSELoss(nn.Module):
    """
    MSELoss with respect to mean and std, a mask can be used to restrict the loss computation to certain regions of the image.
    """
    def forward(self, input, mean_std_target, mask):
        b,c,h,w = input.size()
        b_mask,c_mask,h_mask,w_mask = mask.size()
        assert h == h_mask and w == w_mask, f"image size = ({h},{w}) does not match mask size = ({h_mask}, {w_mask})"
        # duplicate mask along channels and batch dim 
        duplicated_mask = repeat(mask, "1 1 h w -> b c h w", b=b, c=c)
        # calculate masked mean and std from weighted sum of input and target mean/std
        input_mean, input_std = calc_masked_mean_std(input, duplicated_mask)
        out = nn.MSELoss()(input_mean, mean_std_target[0]) + nn.MSELoss()(input_std, mean_std_target[1]) 
        return (out)

def TotalVariationLoss(img):      
    b, c, h, w = img.size()
    tv_h = ((img[:,:,1:,:] - img[:,:,:-1,:]).pow(2)).sum()
    tv_w = ((img[:,:,:,1:] - img[:,:,:,:-1]).pow(2)).sum()    
    return (tv_h + tv_w) / (b * c * h * w)

def TotalVariationAbsLoss(img):      
    b, c, h, w = img.size()
    tv_h = torch.abs((img[:,:,1:,:] - img[:,:,:-1,:])).sum()
    tv_w = torch.abs((img[:,:,:,1:] - img[:,:,:,:-1])).sum()    
    return (tv_h + tv_w) / (b * c * h * w)

#NOTE: different downsampling strategies
def resize_scalar_field(image, out_keys, ceil_mode = False):
    """
    Bilinear downsampling to fit layers of VGG-19.
    """
    out = {}
    out['r11'] = image
    out['r12'] = image 
    if ceil_mode:
        out['r21'] = transforms.Resize(size=((image.shape[2] + 1)//2, (image.shape[3] + 1)//2))(image)
    else:
        out['r21'] = transforms.Resize(size=(image.shape[2] // 2, image.shape[3] // 2))(image)
    out['r22'] = out['r21']
    if ceil_mode:
        out['r31'] = transforms.Resize(size=((image.shape[2] + 3)//4, (image.shape[3] + 3)//4))(image)
    else:
        out['r31'] = transforms.Resize(size=(image.shape[2] // 4, image.shape[3] // 4))(image)
    out['r32'] = out['r31']
    out['r33'] = out['r31']
    out['r34'] = out['r31']
    if ceil_mode:
        out['r41'] = transforms.Resize(size=((image.shape[2] + 7)//8, (image.shape[3] + 7)//8))(image)
    else:
        out['r41'] = transforms.Resize(size=(image.shape[2] // 8, image.shape[3] // 8))(image)
    out['r42'] = out['r41'] 
    out['r43'] = out['r41']
    out['r44'] = out['r41']
    if ceil_mode:
        out['r51'] = transforms.Resize(size=((image.shape[2] + 15)//16, (image.shape[3] + 15)//16))(image)
    else:
        out['r51'] = transforms.Resize(size=(image.shape[2] // 16, image.shape[3] // 16))(image)
    out['r52'] = out['r51'] 
    out['r53'] = out['r51']
    out['r54'] = out['r51']
    return [out[key] for key in out_keys]

def receptive_resize_scalar_field(image, out_keys, use_relu = False, ceil_mode = False):
    """
    Downsampling network with the same structure as VGG-19, i.e. convolutions have unit weights (= Avg. pooling).
    """
    # avg pool instead of conv
    inverse_conv = nn.Sequential(
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.AvgPool2d((3, 3), stride=1)
    )
    max_pool = nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=ceil_mode)
    #NOTE: for min pool instead of max pool
    # max_pool_2x2 = nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=ceil_mode)
    # max_pool = lambda x: -1 * max_pool_2x2(-1 * x)
    relu = nn.ReLU()

    out = {}
    out['r11'] = inverse_conv(image)
    out['r12'] = inverse_conv(out['r11'])
    out['p1'] = max_pool(out['r12'])
    out['r21'] = inverse_conv(out['p1'])
    out['r22'] = inverse_conv(out['r21'])
    out['p2'] = max_pool(out['r22'])
    out['r31'] = inverse_conv(out['p2'])
    out['r32'] = inverse_conv(out['r31'])
    out['r33'] = inverse_conv(out['r32'])
    out['r34'] = inverse_conv(out['r33'])
    out['p3'] = max_pool(out['r34'])
    out['r41'] = inverse_conv(out['p3'])
    out['r42'] = inverse_conv(out['r41'])
    out['r43'] = inverse_conv(out['r42'])
    out['r44'] = inverse_conv(out['r43'])
    out['p4'] = max_pool(out['r44'])
    out['r51'] = inverse_conv(out['p4'])
    out['r52'] = inverse_conv(out['r51'])
    out['r53'] = inverse_conv(out['r52'])
    out['r54'] = inverse_conv(out['r53'])
    out['p5'] = max_pool(out['r54'])

    if use_relu:
        return [relu(out[key]) if key[0] == "r" else out[key] for key in out_keys]
    return [out[key] for key in out_keys]

def minpool_resize_scalar_field(image, out_keys, use_relu = False, ceil_mode = False):
    """
    Minpool downsampling network with the same structure as VGG-19, i.e. convolutions and pooling layers are replaced by min pooling layers
    """
    # min pool instead of conv
    max_pool_3x3 = nn.Sequential(
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.MaxPool2d((3, 3), 1, (0, 0), ceil_mode=ceil_mode)
    )
    inverse_conv = lambda x: -1 * max_pool_3x3(-1 * x)
    
    max_pool_2x2 = nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=ceil_mode)
    min_pool = lambda x: -1 * max_pool_2x2(-1 * x)
    relu = nn.ReLU()

    out = {}
    out['r11'] = inverse_conv(image)
    out['r12'] = inverse_conv(out['r11'])
    out['p1'] = min_pool(out['r12'])
    out['r21'] = inverse_conv(out['p1'])
    out['r22'] = inverse_conv(out['r21'])
    out['p2'] = min_pool(out['r22'])
    out['r31'] = inverse_conv(out['p2'])
    out['r32'] = inverse_conv(out['r31'])
    out['r33'] = inverse_conv(out['r32'])
    out['r34'] = inverse_conv(out['r33'])
    out['p3'] = min_pool(out['r34'])
    out['r41'] = inverse_conv(out['p3'])
    out['r42'] = inverse_conv(out['r41'])
    out['r43'] = inverse_conv(out['r42'])
    out['r44'] = inverse_conv(out['r43'])
    out['p4'] = min_pool(out['r44'])
    out['r51'] = inverse_conv(out['p4'])
    out['r52'] = inverse_conv(out['r51'])
    out['r53'] = inverse_conv(out['r52'])
    out['r54'] = inverse_conv(out['r53'])
    out['p5'] = min_pool(out['r54'])

    if use_relu:
        return [relu(out[key]) if key[0] == "r" else out[key] for key in out_keys]
    return [out[key] for key in out_keys]