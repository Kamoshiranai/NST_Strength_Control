# Modified from https://github.com/czczup/URST (Apache 2.0)
# Original copyright: czczup (Zhe Chen)
# Modifications copyright (c) 2026 Niklas Merk

import os
import torch
from torch import nn
import argparse
from libs.Matrix import MulLayer
from libs.utils import print_options
# from libs.smooth_filter import smooth_filter
from libs.models import encoder3, encoder4, encoder5
from libs.models import decoder3,decoder4, decoder5
from thumb_instance_norm import init_thumbnail_instance_norm
import torchvision.transforms as transforms
import torch.nn.functional as F
import time
from tqdm import tqdm
import numpy as np
from PIL import Image
import math
import einops

import sys
sys.path.append("../..")
sys.path.append("..")
from utils import resize_scalar_field, receptive_resize_scalar_field, minpool_resize_scalar_field, resize_to_max_size, LuminanceRemapper
from tools import unpadding, preprocess
from os.path import splitext, basename


def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def tensor2Image(image):
    image = image.mul_(255.0).add_(0.5).clamp_(0, 255)
    image = image.squeeze(0).permute(1, 2, 0).to(torch.uint8).cpu().numpy()
    image = Image.fromarray(image)
    return image

def style_transfer_high_resolution(patches, scalar_field_patches, sF, padding, save_path, collection=False, save=True):
    stylized_patches = []
    init_thumbnail_instance_norm(matrix, collection=collection)

    for (patch_idx, patch) in tqdm(enumerate(patches), total=patches.shape[0]):
        patch = patch.unsqueeze(0).to(device)
        cF = vgg(patch)
        if (args.layer == 'r41'):
            content_feature = cF[args.layer]
            style_feature = sF[args.layer]
        else:
            content_feature = cF
            style_feature = sF

        if args.no_style:
            feature = content_feature
        else:
            feature = matrix(content_feature, style_feature)

        if args.scalar_field != "":
            #NOTE: resize scalar field to fit dimensions of feature space like relu 4_1
            if args.downsampling_type == "bilinear":
                scalar_field_patch = resize_scalar_field(scalar_field_patches[patch_idx], [args.layer])[0]
            elif args.downsampling_type == "minpool":
                scalar_field_patch = minpool_resize_scalar_field(scalar_field_patches[patch_idx], [args.layer], use_relu = args.downsampling_use_relu)[0]
            elif args.downsampling_type == "receptive":
                scalar_field_patch = receptive_resize_scalar_field(scalar_field_patches[patch_idx], [args.layer], use_relu = args.downsampling_use_relu)[0]
            scalar_field_patch = einops.repeat(scalar_field_patch, "1 h w -> c h w", c=feature.shape[1])
            scalar_field_patch = einops.repeat(scalar_field_patch, "c h w -> b c h w", b = feature.shape[0]) # add batch dim
            scalar_field_patch = scalar_field_patch.to(device)

            #NOTE: interpolate using the scalar field, cut to this patch
            feature = scalar_field_patch * feature + (1.0 - scalar_field_patch) * content_feature
        
        stylized = dec(feature)
        stylized = F.interpolate(stylized, patch.shape[2:], mode='bilinear', align_corners=True)
        stylized = unpadding(stylized, padding=padding)
        stylized_patches.append(stylized.cpu())

    stylized_patches = torch.cat(stylized_patches, dim=0)
    b, c, h, w = stylized_patches.shape
    stylized_patches = stylized_patches.unsqueeze(dim=0)
    stylized_patches = stylized_patches.view(1, b, c * h * w).permute(0, 2, 1).contiguous()
    output_size = (int(math.sqrt(b) * h), int(math.sqrt(b) * w))
    stylized_image = F.fold(stylized_patches, output_size=output_size, kernel_size=(h, w), stride=(h, w))
    if args.print:
        print("stylized image:", stylized_image.shape)
    #NOTE try to fix misaligned image sizes by just cropping pixels from bottom and right
    stylized_image = stylized_image[:, :, :IMAGE_HEIGHT, :IMAGE_WIDTH]

    #NOTE remap colors to content
    output_image = tensor2Image(stylized_image)
    if not args.no_remap_colors:
        output_image = color_remapper.unshift(output_image)
    if save:
        output_image.save(save_path)
        print("stylized image saved to: ", save_path)
    
def style_transfer_thumbnail(thumb, sF, save_path, save=False):
    cF = vgg(thumb)
    #NOTE: this is the output of relu 4_1 or relu 3_1
    init_thumbnail_instance_norm(matrix, collection=True)
    if (args.layer == 'r41'):
        if args.no_style:
            feature = cF[args.layer]
        else:
            feature = matrix(cF[args.layer], sF[args.layer])
    else:
        if args.no_style:
            feature = cF
        else:
            feature = matrix(cF, sF)
    stylized_thumb = dec(feature)
    if save and args.save_thumb:
        stylized_thumb_image = tensor2Image(stylized_thumb)
        stylized_thumb_image.save(save_path)
        print("stylized thumbnail saved to: ", save_path)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--vgg_dir", default='models/vgg_r41.pth',
                        help='pre-trained encoder path')
    parser.add_argument("--decoder_dir", default='models/dec_r41.pth',
                        help='pre-trained decoder path')
    parser.add_argument("--matrixPath", default='models/r41.pth',
                        help='pre-trained model path')
    parser.add_argument("--style", default="../../images/styles/mondrian.jpg",
                        help='path to style image')
    parser.add_argument("--content", default="../../images/content/471.jpg",
                        help='path to frames')
    parser.add_argument("--output", default="../../images/output/urst/",
                        help='path to transferred images')
    parser.add_argument("--layer", default="r41",
                        help='which features to transfer, either r31 or r41')
    parser.add_argument('--patch_size', type=int, default=1000, help='patch size') #NOTE: might need to make sure that patchsize is a mutiple of 8 (else their size gets changed via encoding + decoding, meaning luminance remapping and scalar field downsampling wont work)
    parser.add_argument('--thumb_size', type=int, default=1024, help='thumbnail size')
    parser.add_argument('--style_size', type=int, default=512, help='style size') #NOTE: resize + crop to square, style image to this size
    parser.add_argument('--padding', type=int, default=32, help='padding')
    parser.add_argument('--test_speed', action="store_true", help='test the speed')
    parser.add_argument('--URST', action="store_true", help='use URST framework')
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument('--content_size', type=int, default=0, help='content_size') #NOTE: if given, resizes content to this resolution

    parser.add_argument('--no_style', action="store_true", help='do not stylize content') #NOTE: just pipe content through encoder + decoder without style
    parser.add_argument('--no_remap_colors', action="store_true", help='do not remap colors')
    parser.add_argument('--downsampling_type', type=str, default = 'bilinear', choices=['bilinear', 'receptive', 'minpool'], help="type of downsampling to apply to scalar field before passing through vgg")
    parser.add_argument('--downsampling_use_relu', action="store_true", help="use relu in downsampling layer")
    parser.add_argument('--save_thumb', action="store_true", help='save thumbnail image')
    parser.add_argument('--print', action="store_true", help='do execute print commands')
    parser.add_argument("--scalar_field", default="",
                        help='path to scalar field (should be same size as content, else is resized)')
    parser.add_argument('--invert_scalar_field', action="store_true", help='whether to invert the scalar field')
    parser.add_argument("--content_mask", default="",
                        help='path to content mask used to restrict region of content image for luminance remapping (should be same size as content, else is resized), e.g. if using a texture map that is not rectangular.')

    ################# PREPARATIONS #################
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    os.makedirs(args.output, exist_ok=True)
    content_name = args.content.split("/")[-1].split(".")[0]
    style_name = args.style.split("/")[-1].split(".")[0]
    device = torch.device(args.device)

    ################# MODEL #################
    if(args.layer == 'r31'):
        args.vgg_dir = "models/vgg_r31.pth"
        args.decoder_dir = "models/dec_r31.pth"
        args.matrixPath = "models/r31.pth"
        vgg = encoder3().to(device)
        dec = decoder3().to(device)
    elif(args.layer == 'r41'):
        vgg = encoder4().to(device)
        dec = decoder4().to(device)
    if args.print:
        print_options(args, save_to_file=False)
    matrix = MulLayer(args.layer).to(device)
    vgg.load_state_dict(torch.load(args.vgg_dir, weights_only=True))
    dec.load_state_dict(torch.load(args.decoder_dir, weights_only=True))
    matrix.load_state_dict(torch.load(args.matrixPath, weights_only=True))
    
    PATCH_SIZE = args.patch_size
    PADDING = args.padding
    
    content_tf = test_transform(0, False)
    style_tf = test_transform(args.style_size, True)

    repeat = 15 if args.test_speed else 1
    time_list = []

    for i in range(repeat):
        content_image = Image.open(args.content).convert("RGB")
        if args.content_size != 0:
            content_image = content_image.resize((args.content_size, args.content_size))

        #NOTE: need to make sure content size is a multiple of 8 for luminance remapping and scalar field downsampling with a network to work
        if args.scalar_field != "" or not args.no_remap_colors:
            max_content_size = max(content_image.size[0], content_image.size[1])
            content_image = resize_to_max_size(content_image, max_content_size, multiple_of_8=True)

        if args.scalar_field != "":
            scalar_field_image = Image.open(args.scalar_field).convert("L")
            scalar_field_image = scalar_field_image.resize(content_image.size)
        if args.content_mask != "":
            content_mask_image = Image.open(args.content_mask).convert("L")
            content_mask_image = content_mask_image.resize(content_image.size)
            content_mask = np.array(content_mask_image).astype(np.bool)
        else:
            content_mask = None
        IMAGE_WIDTH, IMAGE_HEIGHT = content_image.size
        if args.print:
            print("image size:", content_image.size)
        style_image = Image.open(args.style)

        if not args.no_remap_colors:
            #NOTE: luminance remapping
            color_remapper = LuminanceRemapper(content_image, content_mask = content_mask)
            style_image = color_remapper.shift(style_image)

        torch.cuda.synchronize()
        start_time = time.time()
        
        if args.URST:
            aspect_ratio = IMAGE_WIDTH / IMAGE_HEIGHT
            thumbnail = content_image.resize((int(aspect_ratio * args.thumb_size), args.thumb_size))
            patches = preprocess(content_image, padding=PADDING, patch_size=PATCH_SIZE, transform=content_tf, cuda=False) # output has size [c, num patches, h, w] where h and w are smaller than patch size + 2 * padding

            if args.scalar_field != "":
                # cut same sized patches out of scalar field
                scalar_field_patches = preprocess(scalar_field_image, padding=PADDING, patch_size=PATCH_SIZE, transform=content_tf, cuda=False)
                # normalize scalar field #TODO this does not restrict the min and max to regions given by a content_mask
                scalar_field_patches -= scalar_field_patches.min()
                scalar_field_patches /= (scalar_field_patches.max() + 1e-6)
                if args.invert_scalar_field:
                    scalar_field_patches = 1.0 - scalar_field_patches
            else:
                scalar_field_patches = torch.empty(0)

            thumbnail = content_tf(thumbnail).unsqueeze(0).to(device)
            style = style_tf(style_image.convert("RGB")).unsqueeze(0).to(device)
            
            if args.print:
                print("patches:", patches.shape)
                print("thumb:", thumbnail.shape)
                print("style:", style.shape)
            
            with torch.no_grad():
                sF = vgg(style)
                save_name = f"{splitext(basename(args.content))[0]}"
                if args.no_style:
                    save_name += "_no_style"
                else:
                    save_name += f"_{splitext(basename(args.style))[0]}_size={args.style_size}" 
                thumbnail_name = save_name + f"_thumb={args.thumb_size}.jpg"
                if args.scalar_field != "":
                    save_name += f"_{splitext(basename(args.scalar_field))[0]}"
                if args.downsampling_type == "receptive":
                    output_name += "_receptive_downsampling"
                elif args.downsampling_type == "minpool":
                    output_name += "_minpool_downsampling"
                if args.downsampling_use_relu:
                    output_name += "_with_relu"
                if args.no_remap_colors:
                    save_name += "_no_remap_colors"
                if args.layer != "r41":
                    save_name += f"_{args.layer}"
                save_name +=".jpg"

                style_transfer_thumbnail(thumbnail, sF, save=False if args.test_speed else True,
                                         save_path=os.path.join(args.output, thumbnail_name)) #NOTE: colors of saved thumbnail image are not remapped, this should not change the result

                style_transfer_high_resolution(
                    patches, scalar_field_patches, sF, padding=PADDING, collection=False,
                    save_path=os.path.join(args.output, save_name), 
                    save=False if args.test_speed else True
                )
        else:
            image = content_tf(content_image).unsqueeze(0).to(device)
            style = style_tf(style_image.convert('RGB')).unsqueeze(0).to(device)

            with torch.no_grad():
                sF = vgg(style)
                style_transfer_thumbnail(content_image, sF, save=False if args.test_speed else True,
                                         save_path=os.path.join(args.output, "original_result.jpg"))
            
        torch.cuda.synchronize()
        time_list.append(time.time() - start_time)

    if args.print:
        print("time: %.2fs" % np.mean(time_list[-10:]))
        print("Max GPU memory allocated: %.3f GB" % (torch.cuda.max_memory_allocated(device=0) / 1024. / 1024. / 1024.))
        # print("Total memory of the current GPU: %.4f GB" % (torch.cuda.get_device_properties(device=0).total_memory / 1024. / 1024 / 1024))
