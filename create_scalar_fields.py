import argparse
from os.path import basename
from os.path import splitext
import torch
import PIL as pil
from torchvision.utils import save_image
from einops import repeat

parser = argparse.ArgumentParser()

parser.add_argument('--content', type=str, default = 'images/content/471.jpg', 
                    help='File path to the content image')

args = parser.parse_args()

content_image = pil.Image.open(args.content)

# create binary mask
scalar_field = torch.zeros(1, content_image.size[1], content_image.size[0])
scalar_field[:, :, :content_image.size[0]//2] = 1

# assemble output name
output_name = "images/scalar_fields/binary_for_" + splitext(basename(args.content))[0] + ".jpg"

save_image(scalar_field, output_name)

# create gradient
scalar_field = torch.arange(0, content_image.size[0]) / (content_image.size[0]-1)
scalar_field = repeat(scalar_field, "w -> h w", h = content_image.size[1])

# assemble output name
output_name = "images/scalar_fields/gradient_for_" + splitext(basename(args.content))[0] + ".jpg"

save_image(scalar_field, output_name)
