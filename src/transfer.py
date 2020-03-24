import torch.utils.data
from src import const
from src.utils import parse_args_and_merge_const
# from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import PIL
import copy
import time
import os
import numpy as np

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        # cv_image = pil2cv(image)
        # h, w = image.shape[:2]
        w = image.width
        h = image.height
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = image.resize((new_h, new_w), resample=PIL.Image.BICUBIC)

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively

        return img

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):
        # cv_image = pil2cv(image)
        w = image.width #256
        h = image.height #256
        new_h, new_w = self.output_size #224, 224

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image.crop((left, top, left + new_w, top + new_h)) #left, upper, right, lower

        return image

data_transforms = {
    'train': transforms.Compose([Rescale(256),
                                 RandomCrop(224),
                                 transforms.ToTensor()
                                 ]),
    'val': transforms.Compose([Rescale(256),
                               RandomCrop(224),
                               transforms.ToTensor()
                               ]),
}

data_dir = '/home/msl/jinwoo_test/venv/skirt/transferLearning/data/skirt_length/not_pre_process_skirt_length'

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                                              shuffle=True, num_workers=4)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)



def train_model(model, criterion, scheduler, optimizer, learning_rate, num_epochs=50):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            if phase == 'val':
                pred_results = list()
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    sample = {}
                    sample['image'] = inputs
                    outputs = model(sample, "transfer")
                    _, preds = torch.max(outputs['attr_output'], 1)
                    loss = criterion(outputs['attr_output'], labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                if phase == 'val':
                    pred_results.append(preds)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc*100))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                with open("./pred.txt", 'w') as f:
                    for d in pred_results:
                        f.write("%s\n" % d.item())

                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        learning_rate *= const.LEARNING_RATE_DECAY
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc*100))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(best_model_wts, "./best_model.pth")

    return model

if __name__ == '__main__':
    parse_args_and_merge_const()
    if os.path.exists('models') is False:
        os.makedirs('models')

    net = const.USE_NET()
    net.load_state_dict(torch.load(const.save_model_path))

    criterion = nn.CrossEntropyLoss()
    learning_rate = const.LEARNING_RATE
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    for param in net.parameters():
        param.requires_grad = False

    net.attr_fc2 = nn.Linear(1024, len(class_names))
    net = net.to(const.device)

    net = train_model(net, criterion, exp_lr_scheduler, optimizer, const.LEARNING_RATE, num_epochs=50)