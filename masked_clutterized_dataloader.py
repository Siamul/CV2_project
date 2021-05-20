import torch
import numpy as np
import torch.utils.data as data_utl
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import random
import cv2
import json
from detectron2.structures.instances import Instances
from detectron2.structures.masks import BitMasks
from detectron2.structures.boxes import Boxes
#multi_process_override = False
#try:
#    mp.set_start_method('spawn')
#except RuntimeError as re:
#    print("Multi-processing not supported!")
#    print(re)
#    multi_process_override = True
#    pass

class datasetLoader(data_utl.Dataset):

    def __init__(self, dataset, transform = None):

        # Image pre-processing
        self.transform = transform

        # Data loading
        self.data = dataset
        #self.assign_classes()


    def __getitem__(self, index):
        imagePath = self.data[index]
        # Reading of the image
        img = Image.open(imagePath).convert('RGB')
        # Applying transformation
        if self.transform is not None:
            transform_img = self.transform(img)
        else:
            transform_img = img
        img.close()
        return transform_img


    def __len__(self):
        return len(self.data)

class maskedDatasetLoader(data_utl.Dataset):

    def __init__(self, dataset, device, transform = None):

        # Image pre-processing
        self.transform = transform

        # Data loading
        self.data = dataset
        #self.assign_classes()
        self.ToTensor = transforms.ToTensor()
        self.device = device


    def __getitem__(self, index):
        imagePath = self.data[index]
        pathParts = imagePath.split("/")
        #print(pathParts)
        jsonName = pathParts[-1].split(".")[0] + ".json"
        #print(jsonName)
        jsonPathList = pathParts[:-1]
        jsonPathList.append(jsonName)
        #print(jsonPathList)
        jsonPath = os.path.join(*jsonPathList)
        if jsonPathList[0] == '':
            jsonPath = '/'+jsonPath
        #print(jsonPath)
        with open(jsonPath, 'r') as f:
            json_data = json.load(f)
        mask_polygon_str = json_data['shapes'][0]['points']
        #print(mask_polygon_str)
        mask_polygon = []
        for i in range(len(mask_polygon_str)):
            mask_polygon.append(list(map(int, mask_polygon_str[i])))
        mask_polygon = np.array(mask_polygon, dtype=np.int32)
        img = self.ToTensor(np.asarray(cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2RGB))).to(self.device)
        mask = np.int32(np.zeros((int(json_data['imageHeight']), int(json_data['imageWidth']))))
        cv2.fillPoly(mask, [mask_polygon], color=(255))
        mask_t = torch.Tensor(mask).to(self.device)
        mask_binary = mask_t / 255
        #print(image.shape)
        #print(mask_binary.shape)
        mask_binary = torch.unsqueeze(mask_binary, 0).to(self.device)
        mask_binary_3d = torch.cat((mask_binary, mask_binary, mask_binary), axis=0)
        # Reading of the image
        #img = Image.open(imagePath).convert('RGB')
        masked_img = img * mask_binary_3d
        masked_img = torch.unsqueeze(masked_img, 0).to(self.device)
        mask_binary_3d = torch.unsqueeze(mask_binary_3d, 0).to(self.device)
        img_mask = torch.cat((masked_img, mask_binary_3d), axis=0)
        # Applying transformation
        if self.transform is not None:
            transform_img_mask = self.transform(img_mask)
        else:
            transform_img_mask = img_mask
        transform_img = transform_img_mask[0,:,:,:].to(self.device)
        transform_mask_binary_3d = transform_img_mask[1,:,:,:].to(self.device)
        return transform_img, transform_mask_binary_3d


    def __len__(self):
        return len(self.data)


class masked_clutterized_datasetLoader:
    def __init__(self, dataset_location, batch_size, device_str="cuda" if torch.cuda.is_available() else "cpu", input_size=800, object_size=512, norm_mean = [0.485, 0.456, 0.406], is_norm = True, norm_var = [0.229, 0.224, 0.225], init_scaling_factor_range = [0.7, 0.95], scaling_factor = 0.95, border=0, data_transform = None, background_transform = None, num_workers=12, is_overlap=False, least_fraction=0.1):
        self.device = torch.device(device_str)
        self.dataset = {}
        self.dataloaders = {}
        self.dataloaders_MP = {}
        for img_folder in os.listdir(dataset_location):
            self.dataset[img_folder] = []
            for img_name in os.listdir(os.path.join(dataset_location, img_folder)):
                if img_name.lower().endswith('.jpg') or img_name.lower().endswith('.png') or img_name.lower().endswith('.jpeg'):
                    self.dataset[img_folder].append(os.path.join(dataset_location, os.path.join(img_folder, img_name)))
        if data_transform is None:
            self.data_transform = transforms.Compose([
                transforms.RandomRotation(180, interpolation = transforms.InterpolationMode.BILINEAR),
                #transforms.RandomHorizontalFlip(),     #reflections at the sides of the totes will be flipped so better not to use these transforms
                #transforms.RandomVerticalFlip(),
                transforms.Resize((object_size,object_size)),
                #transforms.ToTensor()
            ])
        else:
            self.data_transform = data_transform
        
        if background_transform is None:
            self.background_transform = transforms.Compose([
                transforms.Resize((input_size,input_size)),
                transforms.ToTensor()
            ])
        else:
            self.background_transform = background_transform
        self.batch_size = batch_size
        self.dataloader_background = data_utl.DataLoader(dataset = datasetLoader(dataset = self.dataset['background'], transform = self.background_transform),  batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        self.objects = list(self.dataset.keys())
        self.objects.remove('background')
        self.objects.sort()
        self.object_iter = {}
        for object in self.objects:
            if device_str == "cuda":
                self.dataloaders[object] = data_utl.DataLoader(dataset = maskedDatasetLoader(dataset = self.dataset[object], device=self.device, transform = self.data_transform),  batch_size=1, shuffle=True, num_workers=0)
            elif device_str == "cpu":
                self.dataloaders[object] = data_utl.DataLoader(dataset = maskedDatasetLoader(dataset = self.dataset[object], device=self.device, transform = self.data_transform),  batch_size=1, shuffle=True, num_workers=num_workers, pin_memory=True)
            self.object_iter[object] = iter(self.dataloaders[object])            
        self.background_iter = iter(self.dataloader_background)      
        self.init_scaling_factor_range = init_scaling_factor_range
        self.scaling_factor = scaling_factor
        self.border = border
        self.num_workers = num_workers
        #self.thread_executor = f.ThreadPoolExecutor(max_workers = self.num_workers)
        self.process_executor = None
        if device_str == "cpu":
            print("Using CPU, multiprocessing allowed!")
            self.__multi_process_override = False
        else:
            print("Using GPU, multiprocessing not allowed!")
            self.__multi_process_override = True
        if self.__multi_process_override == False:
            from multiprocessing import Pool
            self.process_executor = Pool(self.num_workers)
        self.norm_mean = norm_mean
        self.norm_var = norm_var
        self.is_norm = is_norm
        c_to_i = {}
        for i in range(len(self.objects)):
            c_to_i[self.objects[i]] = i
        self.class_to_idx = c_to_i
        self.is_overlap = is_overlap
        self.least_fraction = least_fraction
    
    def get_idx_to_class(self):   
        return self.objects
    
    def get_class_to_idx(self):
        return self.class_to_idx
    
    def get_num_classes(self):
        return len(self.objects)
    
    def getBatch(self, multi_process = True):
        if multi_process and self.__multi_process_override == False:
            return self.getBatchMP()
        else:
            return self.getBatchSP()  
         
    def getBatchSP(self):
        img_batch = None
        label_batch = None
        mask_batch = None
        objects_batch = []
        try:
            background = next(self.background_iter).to(self.device)
            assert background.shape[0] == self.batch_size
        except Exception as e:
            self.background_iter = iter(self.dataloader_background)
            background = next(self.background_iter).to(self.device)
            assert background.shape[0] == self.batch_size
        for i in range(self.batch_size):
            one_hot_labels = [0] * len(self.objects)
            ins_objects = random.sample(self.objects, random.randint(1, len(self.objects)))
            random.shuffle(ins_objects)
            object_images = []
            object_masks = []
            for object in ins_objects:
                try:
                    object_image, object_mask = next(self.object_iter[object])
                    object_images.append(object_image.to(self.device))
                    object_masks.append(object_mask.to(self.device))
                except Exception as e:
                    self.object_iter[object] = iter(self.dataloaders[object])
                    object_image, object_mask = next(self.object_iter[object])
                    object_images.append(object_image.to(self.device))
                    object_masks.append(object_mask.to(self.device))
            
            bo_tuple = (background[i,:,:,:].to(self.device), object_images, object_masks, ins_objects)
            result = self.getScene(bo_tuple)
            scene_tensor = result[0]
            masks = result[1]
            ins_objects = result[2]
            objects_batch.append(ins_objects)
            for object in ins_objects:
                one_hot_labels[self.objects.index(object)] = 1
            label_t = torch.unsqueeze(torch.Tensor(one_hot_labels).to(self.device),0)
            if label_batch is None:
                label_batch = label_t
            else:
                label_batch = torch.cat((label_batch, label_t), 0)
            if img_batch is None:
                img_batch = scene_tensor.to(self.device)
            else:
                img_batch = torch.cat((img_batch, scene_tensor.to(self.device)), 0)
            if mask_batch is None:
                mask_batch = [masks]
            else:
                mask_batch.append(masks)              
                
        return img_batch, label_batch, objects_batch, mask_batch
        
    def getScene(self, bo_tuple):
        scene = bo_tuple[0] 
        object_images = bo_tuple[1]     
        object_masks = bo_tuple[2]
        ins_objects = bo_tuple[3]   
        channel, height, width = scene.shape
        masks = None
        mask_pix_counts = []
        init_scaling_factor = random.uniform(self.init_scaling_factor_range[0], self.init_scaling_factor_range[1])
        level = len(object_images)
        for i in range(len(object_images)):
            object_img = object_images[i]
            object_mask = object_masks[i]
            scale = init_scaling_factor * (self.scaling_factor ** level)
            level -= 1
            _, _, h, w = object_img.shape
            #if max(h, w) > min(height, width) - self.border*2:
            #    scale *= (min(height, width) - self.border*2) / max(h, w)
            object_transform = transforms.Compose([
                transforms.Resize((int(scale*h), int(scale*w)))
            ])
            object_img = object_transform(object_img)
            object_mask = object_transform(object_mask)
            t_object_img = object_img[0,:,:,:].to(self.device)
            t_object_mask = object_mask[0,:,:,:].to(self.device)
            _, hs, ws = t_object_img.shape
            #print(width, height)
            #print(ws, hs)
            destX = random.randint(self.border, width - ws - self.border)
            destY = random.randint(self.border, height - hs - self.border)
            mask_background = torch.ones((channel, height, width)).to(self.device)
            #print(np_object_img.shape)
            mask_background[:, destY:destY+hs, destX:destX+ws] -= t_object_mask
            mask_latest = torch.ones((channel, height, width)).to(self.device) - mask_background
            masks_new = []
            index_to_be_removed = []
            if masks is None:
                masks = [mask_latest]
                if not self.is_overlap:
                    mask_pix_counts.append(torch.sum(mask_latest))
            else:
                if self.is_overlap:
                    masks.append(mask_latest)
                else:
                    for j in range(len(masks)):
                        old_mask = masks[j]
                        old_mask -= mask_latest
                        old_mask[old_mask < 0] = 0
                        sum_up_old_mask = torch.sum(old_mask)
                        fraction_visible = sum_up_old_mask / mask_pix_counts[j] 
                        if fraction_visible < self.least_fraction:
                            index_to_be_removed.append(j)
                        masks_new.append(old_mask)
                    masks_new.append(mask_latest)
                    mask_pix_counts.append(torch.sum(mask_latest))
                    masks = masks_new
            #scene_mask = torch.ones((channel, height, width)).to(self.device) - mask_background
            #scene *= scene_mask
            scene *= mask_background
            #print(np_object_img.shape)
            scene[:, destY:destY+hs, destX:destX+ws] += t_object_img
        new_masks = []
        new_ins_objects = []
        for i in range(len(ins_objects)):
            if i not in index_to_be_removed:
                new_ins_objects.append(ins_objects[i])
                new_masks.append(masks[i])
                
        #scene = scene[np.newaxis, :, :, :]
        #scene = torch.moveaxis(scene, 0, -1)
        #print(scene.shape)
        scene_tensor = torch.unsqueeze(scene, 0).to(self.device)
        if self.is_norm:
            norm_transform = transforms.Normalize(self.norm_mean, self.norm_var)
            scene_tensor = norm_transform(scene_tensor)
        return (scene_tensor, new_masks, new_ins_objects)                            
        
    '''   
    def getBatchMT(self):
        bo_tuple_list = []
        img_batch = None
        mask_batch = None
        label_batch = None
        objects_batch = None
        try:
            background = next(self.background_iter_MP)
        except Exception as e:
            self.background_iter_MP = iter(self.dataloader_background_MP)
            background = next(self.background_iter_MP).to(self.device)
        for i in range(self.batch_size):
            ins_objects = random.sample(self.objects, random.randint(1, len(self.objects)))
            random.shuffle(ins_objects)
            one_hot_labels = [0] * len(self.objects)
            object_images = []
            for object in ins_objects:
                try:
                    object_images.append(next(self.object_iter[object]).to(self.device))
                except Exception as e:
                    self.object_iter[object] = iter(self.dataloaders[object])
                    object_images.append(next(self.object_iter[object]).to(self.device))
                one_hot_labels[self.objects.index(object)] = 1
            label_t = torch.unsqueeze(torch.FloatTensor(one_hot_labels).to(self.device), 0)
            if label_batch is None:
                label_batch = label_t
            else:
                label_batch = torch.cat((label_batch, label_t), 0)
            if objects_batch is None:
                objects_batch = [ins_objects]
            else:
                objects_batch.append(ins_objects)
            bo_tuple = (background[i,:,:,:].to(self.device), object_images)
            bo_tuple_list.append(bo_tuple)
        results = self.thread_executor.map(self.getScene, bo_tuple_list)
        
        for result in results:
            scene_tensor = result[0]
            masks = result[1] 
            if img_batch is None:
                img_batch = scene_tensor.to(self.device)
            else:
                img_batch = torch.cat((img_batch, scene_tensor.to(self.device)), 0)
            if mask_batch is None:
                mask_batch = [masks]
            else:
                mask_batch.append(masks)
        return img_batch, label_batch, objects_batch, mask_batch
    '''
        
    def getBatchMP(self):
        bo_tuple_list = []
        img_batch = None
        mask_batch = None
        label_batch = None
        objects_batch = None
        try:
            background = next(self.background_iter)
            assert background.shape[0] == self.batch_size
        except Exception as e:
            self.background_iter = iter(self.dataloader_background)
            background = next(self.background_iter)
            assert background.shape[0] == self.batch_size
        for i in range(self.batch_size):
            ins_objects = random.sample(self.objects, random.randint(1, len(self.objects)))
            random.shuffle(ins_objects)
            object_images = []
            object_masks = []
            one_hot_labels = [0] * len(self.objects)
            for object in ins_objects:
                try:
                    object_image, object_mask = next(self.object_iter[object])
                    object_images.append(object_image)
                    object_masks.append(object_mask)
                except Exception as e:
                    self.object_iter[object] = iter(self.dataloaders[object])
                    object_image, object_mask = next(self.object_iter[object])
                    object_images.append(object_image)
                    object_masks.append(object_mask)
                
            bo_tuple = (background[i,:,:,:], object_images, object_masks, ins_objects, self.is_norm, self.norm_mean, self.norm_var, self.init_scaling_factor_range, self.scaling_factor, self.border, self.least_fraction, self.is_overlap)
            bo_tuple_list.append(bo_tuple)
        results = self.process_executor.map(getSceneMP, bo_tuple_list)
        
        for result in results:
            scene_tensor = result[0]
            masks = result[1]
            ins_objects = result[2]
            for object in ins_objects:
                one_hot_labels[self.objects.index(object)] = 1
            label_t = torch.unsqueeze(torch.Tensor(one_hot_labels), 0)
            if label_batch is None:
                label_batch = label_t
            else:
                label_batch = torch.cat((label_batch, label_t), 0)
            if objects_batch is None:
                objects_batch = [ins_objects]
            else:
                objects_batch.append(ins_objects)
            if img_batch is None:
                img_batch = scene_tensor
            else:
                img_batch = torch.cat((img_batch, scene_tensor), 0)
            if mask_batch is None:
                mask_batch = [masks]
            else:
                mask_batch.append(masks)
        return img_batch, label_batch, objects_batch, mask_batch
        
    def getBatchDT2(self):
        img_batch, label_batch, objects_batch, mask_batch = self.getBatch()
        dt2_batch = []
        for i in range(img_batch.shape[0]):
            img_dict = {}
            img_dict['file_name'] = 'File exists in volatile memory'
            img_dict['image_id'] = i
            channel, height, width = img_batch[i].shape
            img_dict['height'] = height
            img_dict['width'] = width
            #print(img_batch[i, :, :])
            img_1 = img_batch[i].to(self.device)
            #print(img_1)
            assert torch.max(img_1) <= 1
            img_255 =  (img_1 * 255).type(torch.uint8).to(self.device)
            #print(img_255)
            img_dict['image'] = img_255
            object_masks = mask_batch[i]
            object_names = objects_batch[i]
            gt_boxes = []
            gt_masks = None
            gt_classes = []
            for j in range(len(object_masks)):
                bitmask = torch.gt(object_masks[j][0], 0).to(self.device)
                rows = torch.any(bitmask, axis=1).to(self.device)
                cols = torch.any(bitmask, axis=0).to(self.device)
                y1, y2 = torch.where(rows)[0][[0, -1]]
                x1, x2 = torch.where(cols)[0][[0, -1]]
                gt_boxes.append([x1, y1, x2, y2])
                if gt_masks is None:
                    gt_masks = torch.unsqueeze(bitmask, 0).to(self.device)
                else:
                    gt_masks = torch.cat((gt_masks, torch.unsqueeze(bitmask, 0).to(self.device)), 0)
                object_name = object_names[j]
                gt_classes.append(self.class_to_idx[object_name])

            gt_boxes = Boxes(torch.tensor(gt_boxes).to(self.device))
            #print('max pixel value: ', torch.max(img_dict['image']), 'min pixel value: ', torch.min(img_dict['image']))
            
            #print('comparing shapes: ', img_batch[i].shape, gt_masks.shape)
            gt_masks = BitMasks(tensor = gt_masks)
            gt_classes = torch.tensor(gt_classes).to(self.device)
            instances = Instances(image_size = (height, width), gt_boxes = gt_boxes, gt_masks = gt_masks, gt_classes = gt_classes)
            img_dict['instances'] = instances
            dt2_batch.append(img_dict)
        return dt2_batch
        
        
        
def getSceneMP(args):
    scene = args[0] 
    object_images = args[1]
    object_masks = args[2]
    ins_objects = args[3]
    is_norm = args[4]
    norm_mean = args[5]
    norm_var = args[6]
    init_scaling_factor_range = args[7]
    scaling_factor = args[8]
    border = args[9]
    least_fraction = args[10]
    is_overlap = args[11]
                 
    channel, height, width = scene.shape
    masks = None
    mask_pix_counts = []
    init_scaling_factor = random.uniform(init_scaling_factor_range[0], init_scaling_factor_range[1])
    level = len(object_images)
    for i in range(len(object_images)):
        object_img = object_images[i]
        object_mask = object_masks[i]
        scale = init_scaling_factor * (scaling_factor ** level)
        level -= 1
        _, _, h, w = object_img.shape
        #if max(h, w) > min(height, width) - self.border*2:
        #    scale *= (min(height, width) - self.border*2) / max(h, w)
        resize_transform = transforms.Resize((int(scale*h), int(scale*w)))
        object_img = resize_transform(object_img)
        object_mask = resize_transform(object_mask)
        t_object_img = object_img[0,:,:,:]
        t_object_mask = object_mask[0,:,:,:]
        _, hs, ws = t_object_img.shape
        #print(width, height)
        #print(ws, hs)
        destX = random.randint(border, width - ws - border)
        destY = random.randint(border, height - hs - border)
        mask_background = torch.ones((channel, height, width))
        #print(np_object_img.shape)
        mask_background[:, destY:destY+hs, destX:destX+ws] -= t_object_mask
        #mask_background[mask_background > 0] = 1
        mask_latest = torch.ones((channel, height, width)) - mask_background
        masks_new = []
        index_to_be_removed = []
        if masks is None:
            masks = [mask_latest]
            if not is_overlap:
                mask_pix_counts.append(torch.sum(mask_latest))
        else:
            if is_overlap:
                masks.append(mask_latest)
            else:
                for j in range(len(masks)):
                    old_mask = masks[j]
                    old_mask -= mask_latest
                    old_mask[old_mask < 0] = 0
                    sum_up_old_mask = torch.sum(old_mask)
                    fraction_visible = sum_up_old_mask / mask_pix_counts[j] 
                    if fraction_visible < least_fraction:
                        index_to_be_removed.append(j)
                    masks_new.append(old_mask)
                masks_new.append(mask_latest)
                mask_pix_counts.append(torch.sum(mask_latest))
                masks = masks_new
        #scene_mask = torch.ones((channel, height, width)).to(self.device) - mask_background
        #scene *= scene_mask
        scene *= mask_background
        #print(np_object_img.shape)
        scene[:, destY:destY+hs, destX:destX+ws] += t_object_img
    #scene = scene[np.newaxis, :, :, :]
    #scene = torch.moveaxis(scene, 0, -1)
    #print(scene.shape)
    new_masks = []
    new_ins_objects = []
    for i in range(len(ins_objects)):
        if i not in index_to_be_removed:
            new_ins_objects.append(ins_objects[i])
            new_masks.append(masks[i])
    scene_tensor = torch.unsqueeze(scene, 0)
    if is_norm:
        norm_transform = transforms.Normalize(norm_mean, norm_var)
        scene_tensor = norm_transform(scene_tensor)
    return (scene_tensor, new_masks, new_ins_objects)   
    
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
            
                

            
                
                
        
        

