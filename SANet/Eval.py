import argparse
import os
import torch
import torch.nn as nn
from torchvision.transforms.functional import InterpolationMode
import numpy as np
import PIL as pil
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image
from einops import repeat, rearrange, pack
import sys
sys.path.append("..")
from utils import resize_scalar_field, receptive_resize_scalar_field, minpool_resize_scalar_field, resize_to_max_size, LuminanceRemapper

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat

def sigmoid(x, center, k=50):
            """Logistic sigmoid centered at 'center'."""
            return 1 / (1 + torch.exp(-k * (x - center)))

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

class SANet(nn.Module):
    
    def __init__(self, in_planes):
        super(SANet, self).__init__()
        self.f = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.g = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.h = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.sm = nn.Softmax(dim = -1)
        self.out_conv = nn.Conv2d(in_planes, in_planes, (1, 1))
        
    def forward(self, content, style):
        F = self.f(mean_variance_norm(content))
        G = self.g(mean_variance_norm(style))
        H = self.h(style)
        b, c, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)
        b, c, h, w = G.size()
        G = G.view(b, -1, w * h)
        S = torch.bmm(F, G)
        S = self.sm(S)
        b, c, h, w = H.size()
        H = H.view(b, -1, w * h)
        O = torch.bmm(H, S.permute(0, 2, 1))
        b, c, h, w = content.size()
        O = O.view(b, c, h, w)
        O = self.out_conv(O)
        O += content
        return O

class Transform(nn.Module):
    def __init__(self, in_planes):
        super(Transform, self).__init__()
        self.sanet4_1 = SANet(in_planes = in_planes)
        self.sanet5_1 = SANet(in_planes = in_planes)
        self.upsample5_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(in_planes, in_planes, (3, 3))
    def forward(self, content4_1, style4_1, content5_1, style5_1):
        sanet4_1 = self.sanet4_1(content4_1, style4_1)
        sanet5_1 = self.sanet5_1(content5_1, style5_1)
        sanet5_1_upsampled = self.upsample5_1(sanet5_1)

        if sanet4_1.shape[2]%2 == 1:
            sanet5_1_upsampled = sanet5_1_upsampled[:, :, :-1, :]
        if sanet4_1.shape[3]%2 == 1:
            sanet5_1_upsampled = sanet5_1_upsampled[:, :, :, :-1]
        
        return self.merge_conv(self.merge_conv_pad(sanet4_1 + sanet5_1_upsampled))

def test_transform():
    transform_list = []
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

parser = argparse.ArgumentParser()

# Basic options
parser.add_argument('--content', type=str, default = '../images/content/471.jpg', 
                    help='File path to the content image')

parser.add_argument('--scalar_field', type=str, default = '',
                    help='File path to greyscale image used as scalar field, should be of same size as content (if not greyscale will be converted according to luminance)')

parser.add_argument('--style', type=str, default = '../images/styles/mondrian.jpg',
                    help='File path to the style image')

parser.add_argument('--second_style', type=str, default = '',
                    help='File path to the second style image, ideally should have same size as first style image')

parser.add_argument("--content_mask", default="",
                        help='path to content mask used to restrict region of content image for luminance remapping (should be same size as content, else is resized), e.g. if using a texture map that is not rectangular.')

parser.add_argument('--no_remap_colors', action=argparse.BooleanOptionalAction, help="don't use luminance-only style transfer")
parser.add_argument('--downsampling_type', type=str, default = 'bilinear', choices=['bilinear', 'receptive', 'minpool'], help="type of downsampling to apply to scalar field before passing through vgg")
parser.add_argument('--downsampling_use_relu', action="store_true", help="use relu in downsampling layer")

parser.add_argument('--steps', type=int, default = 1) #NOTE how often to stylize content image
parser.add_argument('--vgg', type=str, default = 'models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default = 'models/decoder_iter_500000.pth')
parser.add_argument('--transform', type=str, default = 'models/transformer_iter_500000.pth')

parser.add_argument('--strength', type=float, default = 1.0, help="Strength of applied style from 0.0 to 1.0")
parser.add_argument('--style_size', type=int, default = 512, help="Style image is resized s.t. largest extent fits this size")
parser.add_argument('--content_size', type=int, default = 512, help="Content image is resized s.t. largest extent fits this size")

# Additional options
parser.add_argument('--save_ext', default = '.jpg',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default = '../images/output/sanet',
                    help='Directory to save the output image(s)')

# Advanced options

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(args.output):
    os.makedirs(args.output)

decoder = decoder
transform = Transform(in_planes = 512)
vgg = vgg

decoder.eval()
transform.eval()
vgg.eval()

decoder.load_state_dict(torch.load(args.decoder, weights_only=True))
transform.load_state_dict(torch.load(args.transform, weights_only=True))
vgg.load_state_dict(torch.load(args.vgg, weights_only=True))

norm = nn.Sequential(*list(vgg.children())[:1])
enc_1 = nn.Sequential(*list(vgg.children())[:4])  # input -> relu1_1
enc_2 = nn.Sequential(*list(vgg.children())[4:11])  # relu1_1 -> relu2_1
enc_3 = nn.Sequential(*list(vgg.children())[11:18])  # relu2_1 -> relu3_1
enc_4 = nn.Sequential(*list(vgg.children())[18:31])  # relu3_1 -> relu4_1
enc_5 = nn.Sequential(*list(vgg.children())[31:44])  # relu4_1 -> relu5_1

norm.to(device)
enc_1.to(device)
enc_2.to(device)
enc_3.to(device)
enc_4.to(device)
enc_5.to(device)
transform.to(device)
decoder.to(device)

content_tf = test_transform()
style_tf = test_transform()

# NOTE This seems to work for .png or when image is greyscale :)
content_image = pil.Image.open(args.content)
if args.scalar_field != "":
    scalar_field_image = pil.Image.open(args.scalar_field)

style_image = pil.Image.open(args.style)
if args.second_style != "":
    second_style_image = pil.Image.open(args.second_style)

# NOTE: resize s.t. largest dim = content_size
max_style_size = args.style_size
if args.content_size is not None:
    max_content_size = args.content_size
else:
    max_content_size = max(content_image.size[0], content_image.size[1])

content_image = resize_to_max_size(content_image, max_content_size, multiple_of_8=True) # NOTE: content sizes need to be multiple of 8 to be able to combine content color and output luminance channels
style_image = resize_to_max_size(style_image, max_style_size)
if args.second_style != "":
    second_style_image = resize_to_max_size(second_style_image, max_style_size)

if args.scalar_field != "":
    scalar_field_image = scalar_field_image.resize(content_image.size)

if args.content_mask != "":
    content_mask_img = pil.Image.open(args.content_mask).convert("L")
    content_mask_img = content_mask_img.resize(content_image.size)
    content_mask = np.array(content_mask_img).astype(np.bool) # mask is binary
else:
    content_mask = None

if not args.no_remap_colors:
    color_remapper = LuminanceRemapper(content_image, content_mask = content_mask)
    style_image = color_remapper.shift(style_image)
    if args.second_style != "":
        second_style_image = color_remapper.shift(second_style_image)

content = content_tf(content_image.convert("RGB")) # NOTE: transforms.toTensor() gives a float32 tensor with values in range [0,1]

style = style_tf(style_image.convert("RGB"))
if args.second_style != "":
    second_style = style_tf(second_style_image.convert("RGB"))

if args.scalar_field != "":
    scalar_field_image = scalar_field_image.convert("L")
    scalar_field = transforms.ToTensor()(scalar_field_image) #NOTE: has shape (C, H, W)

    # normalize scalar field
    if args.content_mask != "":
        content_mask_torch = rearrange(torch.from_numpy(content_mask), "h w -> 1 h w")
        scalar_field_max = scalar_field[content_mask_torch].max()
        scalar_field_min = scalar_field[content_mask_torch].min()
        scalar_field = (scalar_field - scalar_field_min) / (scalar_field_max - scalar_field_min)
    else:
        scalar_field = (scalar_field - scalar_field.min()) / (scalar_field.max() - scalar_field.min())

    # resize scalar field to size of relu4_1
    if args.downsampling_type == "receptive":
        scalar_field_relu4_1 = receptive_resize_scalar_field(scalar_field, ["r41"], use_relu=args.downsampling_use_relu)[0]
    elif args.downsampling_type == "minpool":
        scalar_field_relu4_1 = minpool_resize_scalar_field(scalar_field, ["r41"], use_relu=args.downsampling_use_relu)[0]
    elif args.downsampling_type == "bilinear":
        scalar_field_relu4_1 = resize_scalar_field(scalar_field, ["r41"])[0]

    scalar_field_relu4_1 = scalar_field_relu4_1.to(device)

content = rearrange(content, "c h w -> 1 c h w").to(device)
styles = rearrange(style, "c h w -> 1 c h w").to(device)
if args.second_style != "":
    second_styles = rearrange(second_style, "c h w -> 1 c h w").to(device)

with torch.no_grad():

    for x in range(args.steps):

        Content4_1 = enc_4(enc_3(enc_2(enc_1(content))))
        Content5_1 = enc_5(Content4_1)
    
        Style4_1 = enc_4(enc_3(enc_2(enc_1(styles))))
        Style5_1 = enc_5(Style4_1)

        if args.second_style != "":
            second_Style4_1 = enc_4(enc_3(enc_2(enc_1(second_styles))))
            second_Style5_1 = enc_5(second_Style4_1)

        Fm_csc = transform(Content4_1, Style4_1, Content5_1, Style5_1) #NOTE: output of transform is F^m_csc
        if args.second_style != "":
            second_Fm_csc = transform(Content4_1, second_Style4_1, Content5_1, second_Style5_1) #NOTE: output of transform is F^m_csc
        Fm_ccc = transform(Content4_1, Content4_1, Content5_1, Content5_1)

        style_strength = args.strength
        if args.scalar_field != "":
            style_strength = 1.0
            scalar_field_relu4_1 *= style_strength
            scalar_field_relu4_1 = repeat(scalar_field_relu4_1, "1 h w -> b c h w", b = Fm_csc.shape[0], c=Fm_csc.shape[1])
            new_Fm_csc = scalar_field_relu4_1 * Fm_csc + (1.0 - scalar_field_relu4_1) * Fm_ccc

        elif args.second_style != "":
            # NOTE: linearly interpolate between two styles
            fade = torch.arange(0, Fm_csc.shape[3], device = device) / (Fm_csc.shape[3]-1)
            fade_mask = repeat(fade, "w -> b c h w", b = Fm_csc.shape[0], h=Fm_csc.shape[2], c=Fm_csc.shape[1])
            new_Fm_csc = fade_mask * Fm_csc + (1.0 - fade_mask) * second_Fm_csc

        # NOTE: uniform style strength
        else:
            new_Fm_csc = style_strength * Fm_csc + (1.0-style_strength) * Fm_ccc

        output = decoder(new_Fm_csc)

        output = output.clamp(0, 1) #NOTE: torch tensors created from PIL images are in range [0,1]

    output = output[0].cpu()

    # assemble output name
    output_name = f'{args.output}/{splitext(basename(args.content))[0]}_size={max_content_size}_stylized_{splitext(basename(args.style))[0]}_size={max_style_size}'

    if args.scalar_field != "":
        output_name += f"_scalar_field_{splitext(basename(args.scalar_field))[0]}"
        if args.downsampling_type == "receptive":
            output_name += "_receptive_downsampling"
        elif args.downsampling_type == "minpool":
            output_name += "_minpool_downsampling"
        if args.downsampling_use_relu:
            output_name += "_with_relu"

    elif args.second_style != "":
        output_name += f"_second_style={splitext(basename(args.second_style))[0]}"

    output_name += f"_strength={style_strength}"

    if args.no_remap_colors:
        output_name += "_no_remap"
    
    output_name += f"{args.save_ext}"
    
    if not args.no_remap_colors:
        # Take color channel from content image
        output_image = transforms.ToPILImage()(output)
        output_image = color_remapper.unshift(output_image)
        output_image.save(output_name, "PNG")
    else:
        save_image(output, output_name) #NOTE: for saving torch tensors as images

    print("image saved to: ", output_name)
