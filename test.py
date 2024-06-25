import torch
from dataset import get_data_transforms, load_data
from torchvision.datasets import ImageFolder
import numpy as np
from torch.utils.data import DataLoader
from dataset import MVTecDataset
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from skimage import measure
import pandas as pd
from numpy import ndarray
from statistics import mean
from scipy.ndimage import gaussian_filter
from sklearn import manifold
from matplotlib.ticker import NullFormatter
from scipy.spatial.distance import pdist
import matplotlib
import pickle
import os

@staticmethod
def cal_anomaly_map(ft_list, fs_list, out_size=[224, 224], use_cos=True, amap_mode='add', gaussian_sigma=0, weights=None):
        # ft_list = [f.cpu() for f in ft_list]
        # fs_list = [f.cpu() for f in fs_list]
        bs = ft_list[0].shape[0]
        weights = weights if weights else [1] * len(ft_list)
        anomaly_map = np.ones([bs] + out_size) if amap_mode == 'mul' else np.zeros([bs] + out_size)
        a_map_list = []
        
        for i in range(len(ft_list)):
            ft = ft_list[i]
            fs = fs_list[i]
            # fs_norm = F.normalize(fs, p=2)
            # ft_norm = F.normalize(ft, p=2)
            if use_cos:
                a_map = 1 - F.cosine_similarity(ft, fs, dim=1)
                a_map = a_map.unsqueeze(dim=1)
            else:
                a_map = torch.sqrt(torch.sum((ft - fs) ** 2, dim=1, keepdim=True))
            a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
            a_map = a_map.squeeze(dim=1)
            a_map = a_map.cpu().detach().numpy()
            a_map_list.append(a_map)
            if amap_mode == 'add':
                anomaly_map += a_map * weights[i]
            else:
                anomaly_map *= a_map
        if amap_mode == 'add':
            anomaly_map /= (len(ft_list) * sum(weights))
                
        if gaussian_sigma > 0:
            for idx in range(anomaly_map.shape[0]):
                anomaly_map[idx] = gaussian_filter(anomaly_map[idx], sigma=gaussian_sigma)
        return anomaly_map, a_map_list


# def cal_anomaly_map(fs_list, ft_list, img, re_img, out_size=224, amap_mode="mul"):
#     if amap_mode == "mul":
#         anomaly_map = np.ones([32, 32])
#     else:
#         anomaly_map = np.zeros([32, 32])
# 
#     a_map_list = []
# 
#     for i in range(len(ft_list) - 1, -1, -1):
#         fs = fs_list[i]
#         ft = ft_list[i]
#         a_map = 1 - F.cosine_similarity(fs, ft)
#         a_map = torch.unsqueeze(a_map, dim=1)
#         a_map = a_map[0, 0, :, :].cpu().detach().numpy()
#         a_map_list.append(a_map)
# 
#     anomaly_map = a_map_list[0]
#     
#     for i in range(1, len(a_map_list)):
#         target_size = a_map_list[i].shape
#         anomaly_map = F.interpolate(torch.from_numpy(anomaly_map).unsqueeze(0).unsqueeze(0), size=target_size, mode="bilinear", align_corners=True)
#         anomaly_map = anomaly_map[0, 0, :, :].cpu().detach().numpy()
#         next_a_map = a_map_list[i]
#         anomaly_map += next_a_map
# 
#     anomaly_map = F.interpolate(torch.from_numpy(anomaly_map).unsqueeze(0).unsqueeze(0), size=out_size, mode="bilinear", align_corners=True)
#     anomaly_map = anomaly_map[0, 0, :, :].cpu().detach().numpy()
# 
#     return anomaly_map, a_map_list


def show_cam_on_image(img, anomaly_map):
    cam = np.float32(anomaly_map) / 255 + np.float32(img) / 255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    if a_min != a_max:
        return (image - a_min) / (a_max - a_min)
    else:
        return image


def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray.transpose(1, 2, 0)), cv2.COLORMAP_JET)
    # gt2 = cv2.cvtColor(gt2.permute(1, 2, 0).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
    return heatmap


def save_image_with_incremental_number(directory, base_name, img, suffix):
    count = 0
    filename = os.path.join(directory, f"{count}_{base_name}_{suffix}.png")
    while os.path.exists(filename):
        count += 1
        filename = os.path.join(directory, f"{count}_{base_name}_{suffix}.png")
    cv2.imwrite(filename, img)


def evaluation(vit_model, dataloader, device, _class_=None):

    vit_model.eval()
    
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    aupro_list = []
    
    with torch.no_grad():
        
        c = 0 #限制視覺化印出來的數量
        for img, gt, label, _ in dataloader:
            img = img.to(device)
            encoder_output, decoder_output = vit_model(img)
            anomaly_map, _ = cal_anomaly_map(encoder_output, decoder_output, out_size=[256, 256])
            # anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            # print(anomaly_map.shape)
            # new_anomaly_map = anomaly_map[np.newaxis, :, :]
            # print(new_anomaly_map.shape)
            
            ##視覺化
            if c < 1:
                
                csv_folder = "/home/server4090-3/Documents/CCC/ADer/results_png"
                os.makedirs(csv_folder, exist_ok=True)

                gt2 = gt
                gt2 = cv2.cvtColor(gt2.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
                gt2 = np.uint8(min_max_norm(gt2) * 255)
                save_image_with_incremental_number('/home/server4090-3/Documents/CCC/ADer/results_png', str(c), gt2, "gt") 
                
                img = cv2.cvtColor(img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
                img = np.uint8(min_max_norm(img) * 255)
                save_image_with_incremental_number('/home/server4090-3/Documents/CCC/ADer/results_png', str(c), img, "org") 
                
                ano_map = min_max_norm(anomaly_map)
                ano_map = cvt2heatmap(ano_map*255)
                ano_map = show_cam_on_image(img, ano_map)
                save_image_with_incremental_number('/home/server4090-3/Documents/CCC/ADer/results_png', str(c), ano_map, "a_map")
                
                # rec_img = cv2.cvtColor(rec_img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
                # rec_img = np.uint8(min_max_norm(rec_img) * 255)
                # save_image_with_incremental_number('/D/CCC/ADer/results_png', str(c), rec_img, "rec")
            
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            if label.item() != 0:
                aupro_list.append(
                    compute_pro(
                        gt.squeeze(0).cpu().numpy().astype(int),
                        anomaly_map,
                    )
                )
            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.ravel())
            gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
            pr_list_sp.append(np.max(anomaly_map))
            
            c += 1

        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 3)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 3)
        
    return auroc_px, auroc_sp, round(np.mean(aupro_list), 3)


def vis_nd(name, _class_):
    print(name, ":", _class_)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    ckp_path = "./checkpoints/" + name + "_" + str(_class_) + ".pth"
    train_dataloader, test_dataloader = load_data(name, _class_, batch_size=16)

    encoder, bn = resnet18(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_resnet18(pretrained=False)
    decoder = decoder.to(device)

    ckp = torch.load(ckp_path)

    decoder.load_state_dict(ckp["decoder"])
    bn.load_state_dict(ckp["bn"])
    decoder.eval()
    bn.eval()

    gt_list_sp = []
    prmax_list_sp = []
    prmean_list_sp = []

    count = 0
    with torch.no_grad():
        for img, label in test_dataloader:
            if img.shape[1] == 1:
                img = img.repeat(1, 3, 1, 1)
            # if count <= 10:
            #    count += 1
            #    continue
            img = img.to(device)
            inputs = encoder(img)
            # print(inputs[-1].shape)
            outputs, rec_img = decoder(bn(inputs))

            anomaly_map, amap_list = cal_anomaly_map(
                inputs, outputs, img.shape[-1], amap_mode="a"
            )
            # anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            ano_map = min_max_norm(anomaly_map)
            ano_map = cvt2heatmap(ano_map * 255)
            img = cv2.cvtColor(
                img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB
            )
            img = np.uint8(min_max_norm(img) * 255)
            cv2.imwrite(
                "./nd_results/"
                + name
                + "_"
                + str(_class_)
                + "_"
                + str(count)
                + "_"
                + "org.png",
                img,
            )
            # plt.imshow(img)
            # plt.axis('off')
            # plt.savefig('org.png')
            # plt.show()
            ano_map = show_cam_on_image(img, ano_map)
            cv2.imwrite(
                "./nd_results/"
                + name
                + "_"
                + str(_class_)
                + "_"
                + str(count)
                + "_"
                + "ad.png",
                ano_map,
            )
            # plt.imshow(ano_map)
            # plt.axis('off')
            # plt.savefig('ad.png')
            # plt.show()

            # gt = gt.cpu().numpy().astype(int)[0][0]*255
            # cv2.imwrite('./results/'+_class_+'_'+str(count)+'_'+'gt.png', gt)

            # b, c, h, w = inputs[2].shape
            # t_feat = F.normalize(inputs[2], p=2).view(c, -1).permute(1, 0).cpu().numpy()
            # s_feat = F.normalize(outputs[2], p=2).view(c, -1).permute(1, 0).cpu().numpy()
            # c = 1-min_max_norm(cv2.resize(anomaly_map,(h,w))).flatten()
            # print(c.shape)
            # t_sne([t_feat, s_feat], c)
            # assert 1 == 2

            # name = 0
            # for anomaly_map in amap_list:
            #    anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            #    ano_map = min_max_norm(anomaly_map)
            #    ano_map = cvt2heatmap(ano_map * 255)
            # ano_map = show_cam_on_image(img, ano_map)
            # cv2.imwrite(str(name) + '.png', ano_map)
            # plt.imshow(ano_map)
            # plt.axis('off')
            # plt.savefig(str(name) + '.png')
            # plt.show()
            #    name+=1
            # count += 1
            # if count>40:
            #    return 0
            # assert 1==2
            gt_list_sp.extend(label.cpu().data.numpy())
            prmax_list_sp.append(np.max(anomaly_map))
            prmean_list_sp.append(
                np.sum(anomaly_map)
            )  # np.sum(anomaly_map.ravel().argsort()[-1:][::-1]))

        gt_list_sp = np.array(gt_list_sp)
        indx1 = gt_list_sp == _class_
        indx2 = gt_list_sp != _class_
        gt_list_sp[indx1] = 0
        gt_list_sp[indx2] = 1

        ano_score = (prmean_list_sp - np.min(prmean_list_sp)) / (
            np.max(prmean_list_sp) - np.min(prmean_list_sp)
        )
        vis_data = {}
        vis_data["Anomaly Score"] = ano_score
        vis_data["Ground Truth"] = np.array(gt_list_sp)
        # print(type(vis_data))
        # np.save('vis.npy',vis_data)
        with open("vis.pkl", "wb") as f:
            pickle.dump(vis_data, f, pickle.HIGHEST_PROTOCOL)


def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:
    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=np.bool_)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        df = pd.concat(
            [
                df,
                pd.DataFrame.from_records(
                    [{"pro": mean(pros), "fpr": fpr, "threshold": th}]
                ),
            ],
            ignore_index=True,
        )

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc


def detection(encoder, bn, decoder, dataloader, device, _class_):
    # _, t_bn = resnet50(pretrained=True)
    bn.load_state_dict(bn.state_dict())
    bn.eval()
    # t_bn.to(device)
    # t_bn.load_state_dict(bn.state_dict())
    decoder.eval()
    gt_list_sp = []
    prmax_list_sp = []
    prmean_list_sp = []
    with torch.no_grad():
        for img, label in dataloader:
            img = img.to(device)
            if img.shape[1] == 1:
                img = img.repeat(1, 3, 1, 1)
            label = label.to(device)
            inputs = encoder(img)
            outputs, rec_img = decoder(bn(inputs))
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], "acc")
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)

            gt_list_sp.extend(label.cpu().data.numpy())
            prmax_list_sp.append(np.max(anomaly_map))
            prmean_list_sp.append(
                np.sum(anomaly_map)
            )  # np.sum(anomaly_map.ravel().argsort()[-1:][::-1]))

        gt_list_sp = np.array(gt_list_sp)
        indx1 = gt_list_sp == _class_
        indx2 = gt_list_sp != _class_
        gt_list_sp[indx1] = 0
        gt_list_sp[indx2] = 1

        auroc_sp_max = round(roc_auc_score(gt_list_sp, prmax_list_sp), 4)
        auroc_sp_mean = round(roc_auc_score(gt_list_sp, prmean_list_sp), 4)
    return auroc_sp_max, auroc_sp_mean
