# CSE 60536 Computer Vision II 
## Semester Project
This repository contains the data and codes for the semester project. 
The midterm project report (contains both Charlie's and my results) can be found here: https://docs.google.com/document/d/1_aEPQlDVgM4MKWv7bmCKboZAt88hgzc8Ua5nBMiE4wM/

The final project report (contains both Charlie's and my results) can be found here: https://docs.google.com/document/d/1U4k5R0X0ijcY43Fd0yVzeveOgfMTpCDuaeY8Ao72bVU/

I have implemented a pytorch dataloader that returns a batch of images with cluttered objects and corresponding masks each time getBatch() is called. It has three different implementations: i) sequentially on CPU ii) multi-processing on CPU (using multiprocessing.pool in python) and iii) utilizing torch with gpu.

The comparison of the three versions is given below:
```
Time taken for processing sequentially using CPU (1 core of Intel Xeon 2.2 GHz):  14.1540949 s
Time taken for processing parallely using CPU (12 cores of Intel Xeon 2.2 GHz):  6.8344337 s
Time taken for processing using GPU (1 GTX 1080 Ti):  3.7211416 s
```
Use the implementation on masked_clutterized_dataloader.py as this works directly with labelme annotated images. The initial implementation uses cropped out images to create the cluttered object images however, there is a problem in using this approach if the color used to fill the cropped out background matches the color of the object (my mug was black and some of the pixels inside the cup where getting replaced by the background pixel).

Instructions on how to use the Dataloader:

Dataset directory arrangement:
```
Dataset                       # Name of the dataset
├── Object1                   # Folder containing all the images of object 1 together with labelme annotations
├── Object2                   # Folder containing all the images of object 2 together with labelme annotations
...
...
└── background                # Folder containing all the images of the background
```
You have to provide the dataset with each object in a separate folder as well as a background folder containing all the background images you want to use.

Example usage:
```
from masked_clutterized_dataloader import masked_clutterized_datasetLoader

data_loader = masked_clutterized_datasetLoader(dataset_location=dataset_location, device_str=device_str, input_size=input_size, object_size=object_size, batch_size=batch_size, is_norm=is_norm, norm_mean=norm_mean, norm_std=norm_std, is_overlap=is_overlap, least_fraction=least_fraction)
```
* dataset_location _specifies the location of the dataset following the directory structure given above._
* input_size _specifies the size the background image is resized to and thus, represents the size of the output image of cluttered objects._
* object_size _specifies the size the object image is resized to before being overlayed onto the background image; this size must be smaller than input_size._
* batch_size _specifies the number of image with cluttered objects return in a single getBatch() call._
* is_norm _specifies whether the returned set of tensors for the images with cluttered objects are normalized using norm_mean and norm_std or not._
* is_overlap _specifies whether you want the mask of objects to contain the occluded parts of the object as well._
* least_fraction _specifies the minimum fraction of object that must be shown on the image (only works for non-overlap mode)_

You can get a batch of tensors containing the images, one-hot encoded labels of the objects present in the image, a list of objects in the image and the binary masks for the objects in the image using:
```
img_batch, label_batch, objects_batch, masks_batch = data_loader.getBatch()
```
* img_batch _contains the images with cluttered objects._
* label_batch _contains the one-hot encoded label for each image specifying which objects are present._
* objects_batch _contains the list of objects in the image._
* masks_batch _contains the binary masks for the different objects in the image arranged in the same order as the objects_batch._

You can get a detectron2 dict by using:
```
dt2_batch = data_loader.getBatchDT2()
```
Here are some examples of the images produced (images were visualized using the Detectron2 Visualizer module; as I did not explicitly mention colors for each object, the colors are chosen randomly and are therefore not consistent across all the images):

Overlap mode:

![Overlap mode](https://github.com/Siamul/CV2_project/blob/main/sample_annot_images_dt2/10.jpg?raw=true)
![Overlap mode](https://github.com/Siamul/CV2_project/blob/main/sample_annot_images_dt2/5.jpg?raw=true)
![Overlap mode](https://github.com/Siamul/CV2_project/blob/main/sample_annot_images_dt2/8.jpg?raw=true)
![Overlap mode](https://github.com/Siamul/CV2_project/blob/main/sample_annot_images_dt2/1.jpg?raw=true)


Non-overlap mode:

![Non-overlap mode](https://github.com/Siamul/CV2_project/blob/main/sample_annot_images_dt2_no_overlap/2.jpg?raw=true)
![Non-overlap mode](https://github.com/Siamul/CV2_project/blob/main/sample_annot_images_dt2_no_overlap/11.jpg?raw=true)
![Non-overlap mode](https://github.com/Siamul/CV2_project/blob/main/sample_annot_images_dt2_no_overlap/5.jpg?raw=true)
![Non-overlap mode](https://github.com/Siamul/CV2_project/blob/main/sample_annot_images_dt2_no_overlap/8.jpg?raw=true)

To train a Mask RCNN model with the dataloader, run:
```
python maskrcnn_trainer.py --modelYAML "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
```
You can utilize whichever model you want, the list can be found at: https://github.com/facebookresearch/detectron2/tree/master/configs/COCO-InstanceSegmentation

The given training routine will save the state_dict of the model every 500 iterations.






Additional Note: I also provide a python script to scrape images from bing using the search queries: 'totes', 'tote', 'empty tote', 'empty totes', 'box', 'boxes', 'empty box', 'empty boxes', 'container', 'containers', 'empty container', 'empty containers', 'carton', 'cartons', 'empty carton', 'empty cartons' which can be modified within the script)
To scrape images and save it with correct formatting (the dataset should be in ./Dataset/), run:
```
python background_image_scraper.py
sh background_preprocess.sh
```
