import torch
from imagen_pytorch import Unet, Imagen
from torchvision.utils import save_image
import torch
import torchvision
from torchvision import transforms    
from imagen_pytorch import Unet, Imagen, ImagenTrainer
import sys
from pycocotools.coco import COCO
import nltk
nltk.download('punkt')
from torchvision import transforms
from data_loader import get_loader
from get_tensor import get_emb_tensor
import torch.utils.data as data

# unet for imagen

unet1 = Unet(
    dim = 32,
    cond_dim = 512,
    dim_mults = (1, 2, 4, 8),
    num_resnet_blocks = 3,
    layer_attns = (False, True, True, True),
    layer_cross_attns = (False, True, True, True)
)

unet2 = Unet(
    dim = 32,
    cond_dim = 512,
    dim_mults = (1, 2, 4, 8),
    num_resnet_blocks = (2, 4, 8, 8),
    layer_attns = (False, False, False, True),
    layer_cross_attns = (False, False, False, True)
)

# imagen, which contains the unets above (base unet and super resoluting ones)

imagen = Imagen(
    unets = (unet1, unet2),
    image_sizes = (64, 256),
    timesteps = 4000,
    cond_drop_prob = 0.1
).cuda()

# mock images (get a lot of this) and text encodings from large T5

# text_embeds = torch.randn(4, 256, 768).cuda()
# images = torch.randn(4, 3, 256, 256).cuda()
trainer = ImagenTrainer(imagen)


# Define a transform to pre-process the training images.
transform_train = transforms.Compose([ 
    transforms.Resize(256),                          # smaller edge of image resized to 256
    transforms.RandomCrop(224),                      # get 224x224 crop from random location
    transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])

# Set the minimum word count threshold.
vocab_threshold = 8

# Specify the batch size.
batch_size = 10

# Obtain the data loader.
data_loader_train = get_loader(transform=transform_train,
                         mode='train',
                         batch_size=batch_size,
                         vocab_threshold=vocab_threshold,
                         vocab_from_file=False,
                         cocoapi_loc = '/home/negar/Documents/Datasets/coco/')

print(len(data_loader_train))

for epoch in range(1, 51):
    # Need to instantiate trainer here, otherwise I get error "AssertionError: you cannot only train on one unet at a time ... "
    # Error occurs here
    trainer = ImagenTrainer(
        imagen, 
        lr = 1e-5, 
        only_train_unet_number = 1
    )
    
    total_loss = 0
    i= 0
    for step in range(int(len(data_loader_train.dataset)/batch_size)): 
        
        indices = data_loader_train.dataset.get_train_indices()
        new_sampler = data.sampler.SubsetRandomSampler( indices )
        data_loader_train.batch_sampler.sampler = new_sampler    
        images,texts = next(iter(data_loader_train))    

        images = images.to("cuda")
        cap, mask = get_emb_tensor(texts)

        loss = trainer(
            images, 
            text_embeds = cap, 
            text_masks= mask,
            unet_number = 1,  
            max_batch_size = 32
        )
        
        trainer.update(unet_number = 1)
        
        total_loss += loss
        i+= 1
        
        if step % 100 == 0:
            print("Loss : ", total_loss/i)    
    trainer.save('./checkpoint.pt')

# do the above for many many many many steps
# now you can sample an image based on the text embeddings from the cascading ddpm

images = imagen.sample(texts = [
    'a whale breaching from afar'
], cond_scale = 3.)

print(images.shape) # (3, 3, 256, 256)
img1 = images[0]
save_image(img1, 'img1.png')


