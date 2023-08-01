import torch
import matplotlib.pyplot as plt
import matplotlib.cm as CM
from tqdm import tqdm

from model import CSRNet
from dataset import CrowdDataset
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
from PIL import Image,ImageDraw,ImageFont
from torchvision import transforms
from pathlib import Path
import numpy as np
from utils import denormalize
import csv
import os
from config import Config
import time
def visual_counting(img,gt, dmap, output,itr):

    img = Image.fromarray((img * 255).astype(np.uint8))
    w, h = img.size
    offset = 204
    max_h, max_w = int(np.ceil(h /offset) + 1), int(np.ceil(w / offset) + 1)
    draw = ImageDraw.Draw(img)
    precision = 0
    recall  = 0
    T_p_all,F_p_all,F_n_all =0,0,0
    for j in range(1, max_h):
        for i in range(1, max_w):



            new_h = j * offset
            new_w = i * offset
            new_x = offset * (i - 1)
            new_y = offset * (j - 1)

            box = [(new_x, new_y),( new_w, new_h)]
           # print(box)
            s_gt = gt[new_y:new_h,new_x:new_w]
            s_pred = dmap[new_y:new_h, new_x:new_w]
            #Precision = (True Positive)/(True Positive + False Positive)
            #Recall = (True Positive)/(True Positive + False Negative)

            count_patch_pred = int(np.round(np.sum(s_pred)))
            count_patch_gt = int(np.round(np.sum(s_gt)))
            if (count_patch_gt-count_patch_pred) < 0: #_predict more apples or similar
                T_p = count_patch_gt
                F_p = np.abs((count_patch_gt-count_patch_pred))
                F_n = 0
            elif (count_patch_gt-count_patch_pred) >0:#count less
                T_p = count_patch_pred
                F_p = 0
                F_n = np.abs((count_patch_gt - count_patch_pred))
            else:#correct counting
                T_p = count_patch_gt
                F_p = 0
                F_n = 0

            T_p_all += T_p
            F_p_all += F_p
            F_n_all += F_n
           # if T_p != 0 and F_p !=0:
            #    precision +=(T_p)/(T_p + F_p)
            #if T_p != 0 and F_n != 0:
             #   recall += (T_p) / (T_p + F_n)

     
            # Create a Rectangle patch
            draw.rectangle(box)
            font = ImageFont.truetype("NotoSans-Regular.ttf", size=40)
            draw.text((new_x+15, new_y + 1), str(count_patch_gt),fill='blue',font =font )
            draw.text((new_x +15, new_y + 40), str(count_patch_pred), fill='red',font = font)

           # plt.imshow()
           # plt.show()
    img.save(f'{output}/bounding_box_{itr}.png')
    return T_p_all,F_p_all,F_n_all


def _visualize_(img,dmap,output,itr):
    #print(dmap.shape,np.max(np.array(img)))
   # keep the same aspect ratio as an input image
    fig, ax = plt.subplots(figsize=figaspect(1.0 * img.shape[0] / img.shape[1]))
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
    img = Image.fromarray((img* 255).astype(np.uint8))
    # display an image with density map put on top of it
    alphacom = Image.alpha_composite(img.convert('RGBA'), dmap.resize(img.size))
    filename ='overlap_'+str(itr)+'.png'
    alphacom.save(Path(output,filename),format="png")
    plt.close('all')
    
def rmse(y, y_pred):
    return np.sqrt(np.mean(np.square(y - y_pred)))


def cal_mae(img_root,gt_dmap_root,model_param_path,out_path):
    '''
    Calculate the MAE of the test data.
    img_root: the root of test image data.
    gt_dmap_root: the root of test ground truth density-map data.
    model_param_path: the path of specific mcnn parameters.
    '''

    device=torch.device("cuda")
    model=CSRNet()
    model.load_state_dict(torch.load(model_param_path))
    model.to(device)
    dataset=CrowdDataset(img_root,gt_dmap_root,image_size = cfg.image_size,img_transform=ToTensor(), dmap_transform=ToTensor())
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)
    model.eval()
    mae=0
    header = ['image','sigma','gt_count','dmap_csr','rmse', 'T_p','F_p','F_n']


    dir_csv = Path(out_path.replace('/images','') , 'count_res_CSR_15.csv')
    with open(dir_csv, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
    
    with torch.no_grad():
        for i,data in enumerate(tqdm(dataloader)):
            img = data['image'].to(device)
            gt_dmap = data['densitymap'].to(device)
            f_name = data['image_name']
            et_dmap=model(img)
            T_p,F_p,F_n= visual_counting(img.squeeze(0).squeeze(0).permute(1,2,0).cpu().numpy(), gt_dmap.squeeze(0).squeeze(0).cpu().numpy(), et_dmap.squeeze(0).squeeze(0).cpu().numpy(), out_path, i)
            data_csv = [str(f_name), '15',str(np.ceil(gt_dmap.data.sum().item())),
                        str(np.ceil(et_dmap.data.sum().item())),
                        str(rmse(gt_dmap.data.sum().item(),
                                 et_dmap.data.sum().item())),
                        str(T_p),str(F_p),str(F_n)]
            with open(dir_csv, 'a', encoding='UTF8') as f:
                writer = csv.writer(f)
                writer.writerow(data_csv)
            print("Mae image : "+str(i)+' pred: '+str(et_dmap.data.sum().item())+' gt: '+str(gt_dmap.data.sum().item()))
            mae+=abs(et_dmap.data.sum()-gt_dmap.data.sum()).item()

            
            del img,gt_dmap,et_dmap

    print("model_param_path:"+model_param_path+" mae:"+str(mae/len(dataloader)))

def estimate_density_map(img_root,gt_dmap_root,model_param_path,index,output):
    '''
    Show one estimated density-map.
    img_root: the root of test image data.
    gt_dmap_root: the root of test ground truth density-map data.
    model_param_path: the path of specific mcnn parameters.
    index: the order of the test image in test dataset.
    '''
    device=torch.device("cuda")
    model=CSRNet().to(device)
    model.load_state_dict(torch.load(model_param_path))
    dataset=CrowdDataset(img_root,gt_dmap_root,image_size = cfg.image_size,img_transform=ToTensor(), dmap_transform=ToTensor())
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)
    model.eval()
    for i,data in enumerate(dataloader):
        if i<index:
            img = data['image'].to(device)
            gt_dmap = data['densitymap'].to(device)
            
            with torch.no_grad():
                et_dmap=model(img).detach()
            et_dmap=et_dmap.squeeze(0).squeeze(0).cpu().numpy()


            img=img.squeeze(0).squeeze(0).permute(1,2,0).cpu().numpy()
            gt_dmap=gt_dmap.squeeze(0).squeeze(0).cpu().numpy()
            _visualize_(img,et_dmap,output,i)

            plt.imsave(f'{output}/pred_{i}.png',et_dmap,cmap=CM.jet)
            plt.imsave(f'{output}/gt_{i}.png', gt_dmap, cmap=CM.jet)






if __name__=="__main__":
    cfg = Config(phase='test')
    torch.backends.cudnn.enabled=False
    img_root='../dataset/all_data'
    gt_dmap_root='test'
    model_param_path='./checkpoints_weighted_loss/63best_model.pth'
    output = 'test_results_63_weighted_loss'   
    output =os.path.join( output , 'images')
    os.makedirs(output, exist_ok = True)
    os.chmod(output, mode =  0o777)
    cal_mae(img_root,gt_dmap_root,model_param_path,output)
    estimate_density_map(img_root,gt_dmap_root,model_param_path,100000,output)

