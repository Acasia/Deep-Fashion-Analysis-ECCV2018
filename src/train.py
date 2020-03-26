import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import PIL
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from src.dataset import DeepFashionCAPDataset
from src.const import base_path
import pandas as pd
import torch.utils.data
from src import const
from src.utils import parse_args_and_merge_const
from tensorboardX import SummaryWriter
from torchvision import transforms
from PIL import Image
import io




def plot_landmarks(image, landmarks):
    """
    Creates an RGB image with the landmarks. The generated image will be of the same size as the frame where the fashion
    matching the landmarks.
    The image is created by plotting the coordinates of the landmarks using matplotlib, and then converting the
    plot to an image.
    Things to watch out for:
    * The figure where the landmarks will be plotted must have the same size as the image to create, but matplotlib
    only accepts the size in inches, so it must be converted to pixels using the DPI of the screen.
    * A white background is printed on the image (an array of ones) in order to keep the figure from being flipped.
    * The axis must be turned off and the subplot must be adjusted to remove the space where the axis would normally be.
    :param frame: Image with a Fashion matching the landmarks.
    :param landmarks: Landmarks of the provided frame, torch.Size([32, 8, 224, 224])
    :return: RGB image with the landmarks as a Pillow Image.
    """

    batch_size, channel_num, image_h, image_w = image.size()

    np_image = np.ones((image_h, image_w, channel_num))
    np_image[:, :, :] = image[0].permute(1, 2, 0).cpu().numpy()

    fig, ax = plt.subplots()
    ax.imshow(np_image, extent=[0, 1, 0, 1])

    for point in range(landmarks.shape[1]):
        ax.plot(landmarks[0][point][0], landmarks[0][point][1], 'ro', markersize=10)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    im = Image.open(buf)
    im.show()
    buf.close()

    pil2tensor = transforms.ToTensor()

    return pil2tensor(im)

if __name__ == '__main__':
    parse_args_and_merge_const()
    if os.path.exists('models') is False:
        os.makedirs('models')

    df = pd.read_csv(base_path + const.USE_CSV)
    train_df = df[df['evaluation_status'] == 'train']
    train_dataset = DeepFashionCAPDataset(train_df, mode=const.DATASET_PROC_METHOD_TRAIN)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=const.BATCH_SIZE, shuffle=True, num_workers=4)
    val_df = df[df['evaluation_status'] == 'val']
    val_dataset = DeepFashionCAPDataset(val_df, mode=const.DATASET_PROC_METHOD_VAL)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=const.VAL_BATCH_SIZE, shuffle=False, num_workers=4)
    val_step = len(val_dataloader)

    net = const.USE_NET()
    net = net.to(const.device)

    learning_rate = const.LEARNING_RATE
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    writer = SummaryWriter(const.TRAIN_DIR)

    total_step = len(train_dataloader)
    step = 0
    best_val_acc = 0

    for epoch in range(const.NUM_EPOCH):
        net.train()
        for i, sample in enumerate(train_dataloader):
            step += 1
            for key in sample:
                sample[key] = sample[key].to(const.device)
            output = net(sample)
            loss = net.cal_loss(sample, output)

            optimizer.zero_grad()
            loss['all'].backward()
            optimizer.step()

            writer.add_image('Image/train_landmark', plot_landmarks(sample['image'], output['lm_pos_output']),
                             epoch)

            if (i + 1) % 10 == 0:
                if 'category_loss' in loss:
                    writer.add_scalar('loss/category_loss', loss['category_loss'], step)
                    writer.add_scalar('loss_weighted/category_loss', loss['weighted_category_loss'], step)
                if 'attr_loss' in loss:
                    writer.add_scalar('loss/attr_loss', loss['attr_loss'], step)
                    writer.add_scalar('loss_weighted/attr_loss', loss['weighted_attr_loss'], step)
                if 'lm_vis_loss' in loss:
                    writer.add_scalar('loss/lm_vis_loss', loss['lm_vis_loss'], step)
                    writer.add_scalar('loss_weighted/lm_vis_loss', loss['weighted_lm_vis_loss'], step)
                if 'lm_pos_loss' in loss:
                    writer.add_scalar('loss/lm_pos_loss', loss['lm_pos_loss'], step)
                    writer.add_scalar('loss_weighted/lm_pos_loss', loss['weighted_lm_pos_loss'], step)
                writer.add_scalar('loss_weighted/all', loss['all'], step)
                writer.add_scalar(  'global/learning_rate', learning_rate, step)
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, const.NUM_EPOCH, i + 1, total_step, loss['all'].item()))
            if (i + 1) % 6000 == 0 and const.VAL_WHILE_TRAIN:
                print('Now Evaluate..')
                with torch.no_grad():
                    net.eval()
                    evaluator = const.EVALUATOR()
                    for j, sample in enumerate(val_dataloader):
                        for key in sample:
                            sample[key] = sample[key].to(const.device)
                        output = net(sample)
                        val_loss = net.cal_loss(sample, output)
                        evaluator.add(output, sample)
                        writer.add_image('Image/val_landmark', plot_landmarks(sample['image'], output['lm_pos_output']),
                                         epoch)
                        if (j + 1) % 100 == 0:
                            print('Val Step [{}/{}]'.format(j + 1, val_step))
                    ret = evaluator.evaluate()
                    writer.add_scalar('loss/val_all_loss', val_loss['all'], step)

                    #category accuracy
                    for topk, accuracy in ret['category_accuracy_topk'].items():
                        print('metrics/category_top{}'.format(topk), accuracy)
                        writer.add_scalar('metrics/category_top{}'.format(topk), accuracy, step)
                        if accuracy > best_val_acc:
                            best_val_acc = accuracy
                            print('Saving Model....')
                            torch.save(net.state_dict(), 'models/' + const.MODEL_NAME)


                    for topk, accuracy in ret['attr_group_recall'].items():
                        for attr_type in range(1, 6):
                            print('metrics/attr_top{}_type_{}_{}_recall'.format(
                                topk, attr_type, const.attrtype2name[attr_type]), accuracy[attr_type - 1]
                            )
                            writer.add_scalar('metrics/attr_top{}_type_{}_{}_recall'.format(
                                topk, attr_type, const.attrtype2name[attr_type]), accuracy[attr_type - 1], step
                            )
                        print('metrics/attr_top{}_all_recall'.format(topk), ret['attr_recall'][topk])
                        writer.add_scalar('metrics/attr_top{}_all_recall'.format(topk), ret['attr_recall'][topk], step)

                    for i in range(8):
                        print('metrics/dist_part_{}_{}'.format(i, const.lm2name[i]), ret['lm_individual_dist'][i])
                        writer.add_scalar('metrics/dist_part_{}_{}'.format(i, const.lm2name[i]), ret['lm_individual_dist'][i], step)
                    print('metrics/dist_all', ret['lm_dist'])
                    writer.add_scalar('metrics/dist_all', ret['lm_dist'], step)
                net.train()
        # learning rate decay
        learning_rate *= const.LEARNING_RATE_DECAY
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

