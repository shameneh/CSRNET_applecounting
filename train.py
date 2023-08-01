#%%
import numpy as np
import time
import torch
import torch.nn as nn
import os
import sys
from tqdm import tqdm

from config import Config
from model import CSRNet

from dataset import create_train_dataloader,create_test_dataloader
from utils import denormalize,weighted_mse_loss
import wandb
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
from PIL import Image
from torchvision import transforms  
def _visualize_(img,dmap):
    #print(dmap.shape,np.max(np.array(img)))
   # keep the same aspect ratio as an input image
    fig,ax= plt.subplots(figsize=figaspect(1.0 * img.shape[0] / img.shape[1]))
    fig.subplots_adjust(0, 0, 1, 1)


    # plot a density map without axis
    ax.imshow(dmap, cmap="jet")
    plt.axis('off')
    fig.canvas.draw()


    # create a PIL image from a matplotlib figure
    dmap = Image.frombytes('RGB',
                           fig.canvas.get_width_height(),
                           fig.canvas.tostring_rgb())

    
    # add a alpha channel proportional to a density map value
    dmap.putalpha(dmap.convert('L'))
    img = Image.fromarray((np.array(img)* 255).astype(np.uint8))
    # display an image with density map put on top of it
    alphacom = Image.alpha_composite(img.convert('RGBA'), dmap.resize(img.size))
    plt.close('all')
    return alphacom

if __name__=="__main__":
    cfg = Config()
    if cfg.lds:
        from dataset_lds import create_train_dataloader,create_test_dataloader
    
    wandb.init(entity="vivid", project="object_counting_dmap", config={"learning_rate": cfg.lr,
                                                                       "architecture": 'CSRNet',
                                                                       "dataset":'train_in_spect',
                                                                       "sigma": str(15),
                                                                       "epoch": cfg.epochs,
                                                                       "batch_size": cfg.batch_size,
                                                                       "flip":True, })  
   
   
    model = CSRNet().to(cfg.device)                                         # model
    wandb.watch(model,log ='all')
    criterion_count = torch.nn.SmoothL1Loss()
    criterion =nn.MSELoss(size_average=False,reduction='sum')                              # objective
    optimizer = torch.optim.Adam(model.parameters(),lr=cfg.lr)              # optimizer
    #for original size  image_size = None 
    if cfg.lds:
        #dataset_lds.py
        #criterion =weighted_mse_loss()  
        train_dataloader = create_train_dataloader(cfg.dataset_root, use_flip=True,image_size = cfg.image_size, batch_size=cfg.batch_size, lds=True, lds_kernel='gaussian', lds_ks=5, lds_sigma=2)
    else:
        train_dataloader = create_train_dataloader(cfg.dataset_root, use_flip=True,image_size = cfg.image_size, batch_size=cfg.batch_size)
    test_dataloader  = create_test_dataloader(cfg.dataset_root,image_size = cfg.image_size)             # dataloader

    min_mae = sys.maxsize
    min_mae_epoch = -1
    for epoch in range(1, cfg.epochs):                          # start training
        model.train()
        epoch_loss = 0.0
        for i, data in enumerate(tqdm(train_dataloader)):
            image = data['image'].to(cfg.device)
            gt_densitymap = data['densitymap'].to(cfg.device)
            et_densitymap = model(image)                        # forward propagation
            if cfg.lds:
                weight = data['weight'].to(cfg.device)
                loss_dmap = weighted_mse_loss(et_densitymap,gt_densitymap,weight)       # calculate loss
                
            else:
                loss_dmap = criterion(et_densitymap,gt_densitymap)       # calculate loss
            true_values_batch, predicted_values_batch = [],[]
            loss = loss_dmap#+loss_count
            optimizer.zero_grad()
            loss.backward()                                     # back propagation
            epoch_loss += loss.item()      
            optimizer.step()                                    # update network parameters
        wandb.log({"train/loss": epoch_loss/len(train_dataloader)})
#        cfg.writer.add_scalar('Train_Loss', epoch_loss/len(train_dataloader), epoch)

        model.eval()
        example_images ,overlap_images= [],[]
        with torch.no_grad():
            epoch_mae = 0.0
            for i, data in enumerate(tqdm(test_dataloader)):
                image = data['image'].to(cfg.device)
                gt_densitymap = data['densitymap'].to(cfg.device)
                et_densitymap = model(image).detach()           # forward propagation
                mae = abs(et_densitymap.data.sum()-gt_densitymap.data.sum())
                epoch_mae += mae.item()
                # WandB – Log images in your test dataset automatically, along with predicted and true labels by passing pytorch tensors with image data into wandb.Image
                if i <10:
                    example_images.append(wandb.Image(
                        et_densitymap, caption="Pred: {} Truth: {}".format(et_densitymap.cpu().sum(),gt_densitymap.cpu().sum())))                
                    overlap_images.append(wandb.Image(_visualize_(denormalize(image[0].cpu()).squeeze(0).squeeze(0).permute(1,2,0),et_densitymap.squeeze(0).squeeze(0).cpu().numpy()),
                        caption="Pred: {} Truth: {}".format(et_densitymap.cpu().sum(),gt_densitymap.cpu().sum())))
            epoch_mae /= len(test_dataloader)
            if epoch_mae < min_mae:
                min_mae, min_mae_epoch = epoch_mae, epoch
                torch.save(model.state_dict(), os.path.join(cfg.checkpoints,str(epoch)+"best_model.pth"))     # save checkpoints
            #elif epoch%10 ==0:
            #    torch.save(model.state_dict(), os.path.join(cfg.checkpoints,str(epoch)+".pth"))     # save checkpoints
            print('Epoch ', epoch, ' MAE: ', epoch_mae, ' Min MAE: ', min_mae, ' Min Epoch: ', min_mae_epoch)   # print information
  #          cfg.writer.add_scalar('Val_MAE', epoch_mae, epoch)
   #         cfg.writer.add_image(str(epoch)+'/Image', denormalize(image[0].cpu()))
    #        cfg.writer.add_image(str(epoch)+'/Estimate density count:'+ str('%.2f'%(et_densitymap[0].cpu().sum())), et_densitymap[0]/torch.max(et_densitymap[0]))
     #       cfg.writer.add_image(str(epoch)+'/Ground Truth count:'+ str('%.2f'%(gt_densitymap[0].cpu().sum())), gt_densitymap[0]/torch.max(gt_densitymap[0]))
            wandb.log({'overlap_images': overlap_images, 'predicted_dmap': example_images,'Val_MAE': epoch_mae,'Estimate density count': et_densitymap[0].cpu().sum()})
# %%
