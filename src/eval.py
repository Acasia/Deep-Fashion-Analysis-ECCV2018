from src.dataset import DeepFashionCAPDataset
from src.const import base_path
import pandas as pd
import torch
import torch.utils.data
from src import const
from src.utils import parse_args_and_merge_const
import os


if __name__ == '__main__':
    parse_args_and_merge_const()
    if os.path.exists('models') is False:
        os.makedirs('models')

    df = pd.read_csv(base_path + const.USE_CSV)
    test_df = df[df['evaluation_status'] == 'test']
    test_dataset = DeepFashionCAPDataset(test_df, mode=const.DATASET_PROC_METHOD_TRAIN)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=const.BATCH_SIZE, shuffle=False, num_workers=4)
    test_step = len(test_dataloader)

    net = const.USE_NET()
    net.load_state_dict(torch.load(const.save_model_path))
    net = net.to(const.device)

    step = 0
    for epoch in range(const.NUM_EPOCH):
        net.eval()
        with torch.no_grad():
            for i, sample in enumerate(test_dataloader):
                step += 1
                for key in sample:
                    sample[key] = sample[key].to(const.device)
                output = net(sample)
                loss = net.cal_loss(sample, output)
                print(loss)
