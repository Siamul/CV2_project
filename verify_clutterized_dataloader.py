import os
import torch
import cv2
from clutterized_dataloader import clutterized_datasetLoader
import torchvision
import sys
import matplotlib.pyplot as plt

if not os.path.exists('./sample_images'):
    os.mkdir('./sample_images')

device_str = "cuda" if torch.cuda.is_available() else "cpu"
clutterized_dataloader = clutterized_datasetLoader(dataset_location = './Dataset/', device_str=device_str, input_size=800, object_size=512, batch_size=32)

img_batch, label_batch, _, _ = clutterized_dataloader.getBatch()

idx_to_class = clutterized_dataloader.get_idx_to_class()

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

unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

for i in range(len(img_batch)):
    img_t = unorm(img_batch[i])
    label_t = label_batch[i]
    objects_present = ''
    for j in range(len(label_t)):
        if label_t[j] == 1:
            objects_present += idx_to_class[j] + ', '
        if j%5 == 0 and j != 0:
            objects_present += '\n'
    img = torch.moveaxis(img_t, 0, -1).detach().cpu().numpy()
    print(img.shape)
    fig = plt.figure()
    fig.suptitle(objects_present)
    plt.imshow(img)
    fig.savefig('./sample_images/'+str(i)+'.jpg')
    plt.close(fig)
    
    
        


