#-*- coding:utf-8 -*-

import pandas as pd
import os
from tensorboardX import SummaryWriter

import torch.utils.data
import torch

from src import const
from src.dataset import DeepFashionCAPDataset
from src.const import base_path
from src.utils import parse_args_and_merge_const
from src.plot_landmark import plot_landmarks

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np


classes = ['Anorak', 'Blazer', 'Blouse', 'Bomber', 'Button-Down', 'Cardigan', 'Flannel', 'Halter', 'Henley', 'Hoodie',
           'Jacket', 'Jersey', 'Parka', 'Peacoat', 'Poncho', 'Sweater', 'Tank', 'Tee', 'Top', 'Turtleneck', 'Capris',
           'Chinos', 'Culottes', 'Cutoffs', 'Gauchos', 'Jeans', 'Jeggings', 'Jodhpurs', 'Joggers', 'Leggings',
           'Sarong', 'Shorts', 'Skirt', 'Sweatpants', 'Sweatshorts', 'Trunks', 'Caftan', 'Cape', 'Coat', 'Coverup',
           'Dress', 'Jumpsuit', 'Kaftan', 'Kimono', 'Nightdress', 'Onesie', 'Robe', 'Romper', 'Shirtdress', 'Sundress']

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

# 헬퍼 함수
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)

    unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    img = unorm(img)
    npimg = img.cpu().numpy()

    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def plot_classes_preds(preds, images, labels):
    '''
    학습된 신경망과 배치로부터 가져온 이미지 / 라벨을 사용하여 matplotlib
    Figure를 생성합니다. 이는 신경망의 예측 결과 / 확률과 함께 정답을 보여주며,
    예측 결과가 맞았는지 여부에 따라 색을 다르게 표시합니다. "images_to_probs"
    함수를 사용합니다.
    '''
    # 배치에서 이미지를 가져와 예측 결과 / 정답과 함께 표시(plot)합니다

    show_batch_size = 16
    fig = plt.figure(figsize=(10, 10))
    for idx in np.arange(show_batch_size):
        ax = fig.add_subplot(4, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx])
        ax.set_title("{0}, (label: {1})".format(
            classes[preds[idx].max(dim=0)[1].item()],
            classes[int(labels[idx].item())]),
            color=("green" if preds[idx].max(dim=0)[1].item()==int(labels[idx].item()) else "red")
        )
    return fig


if __name__ == '__main__':
    parse_args_and_merge_const()
    if os.path.exists('models') is False:
        os.makedirs('models')

    df = pd.read_csv(base_path + const.USE_CSV)
    test_df = df[df['evaluation_status'] == 'test']
    test_dataset = DeepFashionCAPDataset(test_df, mode=const.DATASET_PROC_METHOD_TRAIN)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=const.BATCH_SIZE, shuffle=True, num_workers=4)
    test_step = len(test_dataloader)

    net = const.USE_NET()
    net.load_state_dict(torch.load(const.save_model_path))
    net = net.to(const.device)

    writer = SummaryWriter(const.VAL_DIR)


    step = 0
    print("Start Evaluate")
    net.eval()
    with torch.no_grad():
        for i, sample in enumerate(test_dataloader):
            step += 1
            for key in sample:
                sample[key] = sample[key].to(const.device)
            output = net(sample)
            loss = net.cal_loss(sample, output)

            writer.add_image('Image/val_landmark', plot_landmarks(sample['image'], output['lm_pos_output']),
                             i)

            writer.add_figure('predictions vs. actuals',
                              plot_classes_preds(output['category_output'], sample['image'], sample['category_label']),
                              i)

            print("Validation : {}/{}".format(i+1, test_step))

