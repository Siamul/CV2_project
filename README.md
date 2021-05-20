# CSE 60536 Computer Vision II 
## Semester Project
This repository contains the data and codes for the semester project. 
The midterm project report (contains both Charlie's and my results) can be found here: https://docs.google.com/document/d/1_aEPQlDVgM4MKWv7bmCKboZAt88hgzc8Ua5nBMiE4wM/
The final project report (contains both Charlie's and my results) can be found here: https://docs.google.com/document/d/1U4k5R0X0ijcY43Fd0yVzeveOgfMTpCDuaeY8Ao72bVU/

Use the implementation on masked_clutterized_dataloader.py as this works directly with labelme annotated images. The initial implementation uses cropped out images to create the cluttered object images however, there is a problem in using this approach if the color used to fill the cropped out background matches the color of the object (my mug was black and some of the pixels inside the cup where getting replaced by the background pixel).

Instructions on how to use the Dataloader:

Dataset directory arrangement:

Dataset                       # Name of the dataset
├── Object1                   # Folder containing all the images of object 1 together with labelme annotations
├── Object2                   # Folder containing all the images of object 2 together with labelme annotations
...
...
└── background                # Folder containing all the images of the background

You have to provide the dataset with each object in a separate folder as well as a background folder containing all the background images you want to use.

Example usage:
```
from masked_clutterized_dataloader import masked_clutterized_datasetLoader

data_loader = masked_clutterized_datasetLoader(dataset_location=dataset_location, device_str=device_str, input_size=input_size, object_size=object_size, batch_size=batch_size, is_norm=is_norm, norm_mean=norm_mean, norm_std=norm_std)
```
* dataset_location _specifies the location of the dataset following the directory structure given above_
* input_size _specifies the size the background image is resized to and thus, represents the size of the output image of cluttered objects
* object_size _specifies the size the object image is resized to before being overlayed onto the background image; this size must be smaller than input_size_\
* batch_size _specifies the number of image with cluttered objects return in a single getBatch() call.
* is_norm _specifies whether the returned set of tensors for the images with cluttered objects are normalized using norm_mean and norm_std or not.

You can get a batch of tensors containing the images, one-hot encoded labels of the objects present in the image, a list of objects in the image and the binary masks for the objects in the image using:
```
img_batch, label_batch, objects_batch, masks_batch = data_loader.getBatch()
```
* img_batch _contains the images with cluttered objects
* label_batch _contains the one-hot encoded label for each image specifying which objects are present
* objects_batch _contains the list of objects in the image
* masks_batch _contains the binary masks for the different objects in the image arranged in the same order as the objects_batch

You can get a detectron2 dict by using:
```
dt2_batch = data_loader.getBatchDT2()
```
To train a Mask RCNN model with the dataloader, run:
```
python maskrcnn_trainer.py --modelYAML "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
```
You can utilize whichever model you want, the list can be found at: https://github.com/facebookresearch/detectron2/tree/master/configs/COCO-InstanceSegmentation
