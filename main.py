import torch
from dataset import get_data_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
from torch.utils.data import DataLoader

from dataset import MVTecDataset
import torch.backends.cudnn as cudnn
from argparse import Namespace as _Namespace
from test import evaluation
from torch.nn import functional as F

import sys
sys.path.append('/home/server4090-3/Documents/CCC/ADer')
from model.vitad import vitad

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import csv
import matplotlib.pyplot as plt
import os
from datetime import datetime
import pytorch_ssim

import torchvision.models as models
from torchsummary import summary



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def loss_fucntion(a, b):
    mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    ssim_loss = pytorch_ssim.SSIM()
    loss = 0
    # loss_ssim = 1 - ssim_loss(img, re_img)
    # loss_mse = mse_loss(img, re_img)
    # loss_cos = torch.mean(1 - cos_loss(img.view(img.shape[0], -1), re_img.view(re_img.shape[0], -1)))
    # loss += 0.9*loss_ssim #設定權重
    
    for item in range(len(a)):
        loss += torch.mean(
            1
            - cos_loss(
                a[item].view(a[item].shape[0], -1), b[item].view(b[item].shape[0], -1)
            )
        )
        
    return loss


def train(_class_):
    
    print(_class_)
    epochs = 100
    learning_rate = 0.001 #0.005
    batch_size = 16
    image_size = 256
    
    ###存每個epoch的結果
    # ckp_dir = "/home/server4090-3/Documents/CCC/新案/RD2/results"
    # csv_filename = _class_ + "_results_0.9_0803.csv"
    # csv_file = os.path.join(ckp_dir, csv_filename)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #print(device)

    ###讀資料、前處理
    data_transform, gt_transform = get_data_transforms(image_size, image_size)
    train_path = "/home/server4090-3/Documents/CCC/新案/dataset_mvtec/" + _class_ + "/train"
    test_path = "/home/server4090-3/Documents/CCC/新案/dataset_mvtec/" + _class_
    ckp_path = "/home/server4090-3/Documents/CCC/ADer/checkpoints/" + "vitad_" + _class_ + ".pth"
    train_data = ImageFolder(root=train_path, transform=data_transform)
    test_data = MVTecDataset(
        root=test_path,
        transform=data_transform,
        gt_transform=gt_transform,
        phase="test",
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_data, batch_size=1, shuffle=False
    )
    
    model_t = _Namespace()
    model_t.name = 'vit_base_patch16_224_dino'
    model_t.kwargs = dict(pretrained=True, checkpoint_path='', strict=True, img_size=image_size, teachers=[3, 6, 9], neck=[12])
    model_f = _Namespace()
    model_f.name = 'fusion'
    model_f.kwargs = dict(pretrained=False, checkpoint_path='', strict=False, dim=768, mul=1)
    model_s = _Namespace()
    model_s.name = 'de_vit_base_patch16_224_dino'
    model_s.kwargs = dict(pretrained=False, checkpoint_path='', strict=False, img_size=image_size, students=[3, 6, 9], depth=9)
    model = _Namespace()
    model.name = 'vitad'
    model.kwargs = dict(pretrained=False, checkpoint_path='', strict=True, model_t=model_t, model_f=model_f, model_s=model_s)
    
    vit_model = vitad(pretrained=False, model_t=model_t, model_f=model_f, model_s=model_s).cuda()

    #優化器
    optimizer = torch.optim.Adam(
        list(vit_model.net_s.parameters()) + list(vit_model.net_fusion.parameters()),
        lr=learning_rate,
        betas=(0.9, 0.999),
    )

    #early stop用
    best_loss = float('inf')
    patience = 5  # 設定容忍的迭代次數
    early_stop_counter = 0  # 計數器，用於記錄持續上升的迭代次數
    
    for epoch in range(epochs):
        vit_model.eval()
        loss_list = []
        for img, label in train_dataloader:
            img = img.to(device)
            encoder_output, decoder_output = vit_model(img)
            loss= loss_fucntion(encoder_output, decoder_output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        
        epoch_loss = np.mean(loss_list)
        
        print("epoch [{}/{}], loss:{:.4f}".format(epoch + 1, epochs, np.mean(loss_list)))
        
        #檢查是否滿足early stop的條件
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print("Early stopping triggered. Training stopped at epoch", epoch)
            auroc_px, auroc_sp, aupro_px = evaluation(
                vit_model, test_dataloader, device
            )
            print(
                "Pixel Auroc:{:.3f}, Sample Auroc:{:.3f}, Pixel Aupro:{:.3}".format(
                    auroc_px, auroc_sp, aupro_px
                )
            )
            torch.save(
                {"vit_model": vit_model.state_dict()}, ckp_path
            )
            break
        
        ###每十次印出一次test結果
        if (epoch + 1) % 10 == 0:
            auroc_px, auroc_sp, aupro_px = evaluation(
                vit_model, test_dataloader, device
            )
            print(
                "Pixel Auroc:{:.3f}, Sample Auroc{:.3f}, Pixel Aupro{:.3}".format(
                    auroc_px, auroc_sp, aupro_px
                )
            )
            torch.save(
                {"vit_model": vit_model.state_dict()}, ckp_path
            )
            
            ###把每次結果額外存起來
            # with open(csv_file, mode="a", newline="") as file:
            #     writer = csv.writer(file)
            #     writer.writerow([epoch + 1, auroc_px, auroc_sp, aupro_px])
            
    return auroc_px, auroc_sp, aupro_px

if __name__ == "__main__":
    setup_seed(111)
    item_list = [
        "carpet",
        "bottle",
        "hazelnut",
        "leather",
        "cable",
        "capsule",
        "grid",
        "pill",
        "transistor",
        "metal_nut",
        "screw",
        "toothbrush",
        "zipper",
        "tile",
        "wood",
    ]
    
    auc_results = []
    
    for i in item_list:
        auroc_px, auroc_sp, aupro_px = train(i)
        auc_results.append([i, auroc_px, auroc_sp, aupro_px])

        csv_folder = "/home/server4090-3/Documents/CCC/ADer/results_vitad"
        os.makedirs(csv_folder, exist_ok=True)
        csv_file = os.path.join(csv_folder, "auc_results_cos_0619_large.csv")
        # csv_file = os.path.join(csv_folder, "auc_results_cos_0619_basedino_100.csv")
        with open(csv_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Class", "Pixel AUC", "Sample AUC", "Pixel AUPR"])
            writer.writerows(auc_results)