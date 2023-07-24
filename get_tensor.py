#@title Define and import some functions
#funny fact, its can fit in my 1060 with 3gb vram and i can run training locally
#also its uses t5-base
import math

from imagen_pytorch.t5 import t5_encode_text
import torch, glob
from imagen_pytorch import Unet, Imagen, ImagenTrainer
from tqdm.notebook import tqdm
import torchvision
from pathlib import Path
import torchvision.transforms as T

def get_emb_tensor(texts):
    # for i in files:
    #   f = open(i, "r")
    #   texts.append(f.read())
    #   f.close()
    text_embeds, text_masks = t5_encode_text(texts,return_attn_mask=True)
    text_embeds, text_masks = map(lambda t: t.to('cuda:0'), (text_embeds, text_masks))
    return text_embeds, text_masks


#thanks KyriaAnnwyn
def get_images_tensor(files):
    img_arr = []
    transforms = torch.nn.Sequential(
        T.Resize([256, 256]),
        T.ConvertImageDtype(torch.float)
    )
    for i in files:
       img_arr.append((transforms(torchvision.io.read_image(i, torchvision.io.ImageReadMode.RGB)) * 2 - 1).unsqueeze(0))
    img_embeds = torch.cat((img_arr), dim=0).to('cuda')
    return img_embeds
