import matplotlib
matplotlib.use('Agg')
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
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