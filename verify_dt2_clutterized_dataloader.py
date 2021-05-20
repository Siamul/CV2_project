from masked_clutterized_dataloader import masked_clutterized_datasetLoader
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from detectron2.utils.visualizer import Visualizer
from detectron2.data.catalog import Metadata
from detectron2.structures.instances import Instances
import cv2

device_str = "cuda" if torch.cuda.is_available() else "cpu"
clutterized_dataloader = masked_clutterized_datasetLoader(dataset_location = './Individual/', device_str=device_str, input_size=512, object_size=384, batch_size=16, is_norm=False)

#nondt2_batch, _ , _ ,_ = clutterized_dataloader.getBatch()
#print(nondt2_batch[0])
dt2_batch = clutterized_dataloader.getBatchDT2()
idx_to_class = clutterized_dataloader.get_idx_to_class()
dataset_metadata = Metadata(name="clutterized_data", thing_classes=idx_to_class)

if not os.path.exists('./sample_images_dt2/'):
    os.mkdir('./sample_images_dt2/')
    
if not os.path.exists('./sample_annot_images_dt2_no_overlap/'):
    os.mkdir('./sample_annot_images_dt2_no_overlap/')

#unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

for i in range(len(dt2_batch)):
    img_t = dt2_batch[i]['image'].clone().detach().cpu().numpy()
    label_t = dt2_batch[i]['instances'].gt_classes
    objects_present = ''
    for j in range(len(label_t)):
        objects_present += idx_to_class[label_t[j]] + ', '
        if j%5 == 0 and j != 0:
            objects_present += '\n'
    img = np.moveaxis(img_t, 0, -1)
    print(img.shape)
    fig = plt.figure()
    fig.suptitle(objects_present)
    plt.imshow(img)
    fig.savefig('./sample_images_dt2/'+str(i)+'.jpg')
    plt.close(fig)
    v = Visualizer(
        img,
        metadata = dataset_metadata,
        scale=0.8
    )
    ds_instances = dt2_batch[i]['instances'].to("cpu")
    instances = Instances(image_size = ds_instances.image_size, pred_boxes = ds_instances.gt_boxes, pred_masks = ds_instances.gt_masks.tensor, pred_classes = ds_instances.gt_classes)
    v = v.draw_instance_predictions(instances)
    cv2.imwrite('./sample_annot_images_dt2_no_overlap/'+str(i)+'.jpg', v.get_image()[:, :, ::-1])
    

