import torch
import numpy as np
import torch.utils.data as data_utl
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import random
import cv2

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


class clutterized_datasetLoader:
    def __init__(self, dataset_location, batch_size, device_str="cuda" if torch.cuda.is_available() else "cpu", input_size=800, object_size=512, init_scaling_factor_range = [0.7, 0.95], scaling_factor = 0.95, border=0, data_transform = None, background_transform = None, num_workers=12):
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
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Resize((object_size,object_size)),
                transforms.ToTensor()
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
        self.object_iter = {}
        for object in self.objects:
            self.dataloaders[object] = data_utl.DataLoader(dataset = datasetLoader(dataset = self.dataset[object], transform = self.data_transform),  batch_size=1, shuffle=True, num_workers=num_workers, pin_memory=True)
            self.object_iter[object] = iter(self.dataloaders[object])            
        self.background_iter = iter(self.dataloader_background)      
        self.init_scaling_factor_range = init_scaling_factor_range
        self.scaling_factor = scaling_factor
        self.border = border
        self.num_workers = num_workers
        #self.thread_executor = f.ThreadPoolExecutor(max_workers = self.num_workers)
        self.process_executor = None
        self.device = torch.device(device_str)
        if device_str == "cpu":
            print("Using CPU, multiprocessing allowed!")
            self.__multi_process_override = False
        else:
            print("Using GPU, multiprocessing not allowed!")
            self.__multi_process_override = True
        if self.__multi_process_override == False:
            from multiprocessing import Pool
            self.process_executor = Pool(self.num_workers)
    
    def get_idx_to_class(self):   
        return self.objects
    
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
        objects_batch = None
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
            object_images = []
            for object in ins_objects:
                try:
                    object_images.append(next(self.object_iter[object]).to(self.device))
                except Exception as e:
                    self.object_iter[object] = iter(self.dataloaders[object])
                    object_images.append(next(self.object_iter[object]).to(self.device))
                one_hot_labels[self.objects.index(object)] = 1
            label_t = torch.unsqueeze(torch.Tensor(one_hot_labels).to(self.device),0)
            if label_batch is None:
                label_batch = label_t
            else:
                label_batch = torch.cat((label_batch, label_t), 0)
            if objects_batch is None:
                objects_batch = [ins_objects]
            else:
                objects_batch.append(ins_objects)
            random.shuffle(ins_objects)
            bo_tuple = (background[i,:,:,:].to(self.device), object_images)
            result = self.getScene(bo_tuple)
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
        
    def getScene(self, bo_tuple):
        scene = bo_tuple[0] 
        object_images = bo_tuple[1]            
        channel, height, width = scene.shape
        masks = None
        init_scaling_factor = random.uniform(self.init_scaling_factor_range[0], self.init_scaling_factor_range[1])
        level = len(object_images)
        for object_img in object_images:
            scale = init_scaling_factor * (self.scaling_factor ** level)
            level -= 1
            _, _, h, w = object_img.shape
            #if max(h, w) > min(height, width) - self.border*2:
            #    scale *= (min(height, width) - self.border*2) / max(h, w)
            object_transform = transforms.Compose([
                transforms.Resize((int(scale*h), int(scale*w)))
            ])
            object_img = object_transform(object_img)
            t_object_img = object_img[0,:,:,:].to(self.device)
            _, hs, ws = t_object_img.shape
            #print(width, height)
            #print(ws, hs)
            destX = random.randint(self.border, width - ws - self.border)
            destY = random.randint(self.border, height - hs - self.border)
            mask_background = torch.zeros((channel, height, width)).to(self.device)
            #print(np_object_img.shape)
            mask_background[:, destY:destY+hs, destX:destX+ws] += t_object_img
            mask_background[mask_background > 0] = 1
            if masks is None:
                masks = [mask_background]
            else:
                masks.append(mask_background)
            scene_mask = torch.ones((channel, height, width)).to(self.device) - mask_background
            scene *= scene_mask
            #print(np_object_img.shape)
            scene[:, destY:destY+hs, destX:destX+ws] += t_object_img
        #scene = scene[np.newaxis, :, :, :]
        #scene = torch.moveaxis(scene, 0, -1)
        #print(scene.shape)
        scene_tensor = torch.unsqueeze(scene, 0).to(self.device)
        final_transform = transforms.Compose([
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        scene_tensor = final_transform(scene_tensor)
        return (scene_tensor, masks)                            
        
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
            background = next(self.background_iter).to(self.device)
            assert background.shape[0] == self.batch_size
        for i in range(self.batch_size):
            ins_objects = random.sample(self.objects, random.randint(1, len(self.objects)))
            random.shuffle(ins_objects)
            object_images = []
            one_hot_labels = [0] * len(self.objects)
            for object in ins_objects:
                try:
                    object_images.append(next(self.object_iter[object]).to(self.device))
                except Exception as e:
                    self.object_iter[object] = iter(self.dataloaders[object])
                    object_images.append(next(self.object_iter[object]).to(self.device))
                one_hot_labels[self.objects.index(object)] = 1
            label_t = torch.unsqueeze(torch.Tensor(one_hot_labels).to(self.device), 0)
            if label_batch is None:
                label_batch = label_t
            else:
                label_batch = torch.cat((label_batch, label_t), 0)
            if objects_batch is None:
                objects_batch = [ins_objects]
            else:
                objects_batch.append(ins_objects)
            bo_tuple = (background[i,:,:,:].to(self.device), object_images, self.init_scaling_factor_range, self.scaling_factor, self.border, self.device)
            bo_tuple_list.append(bo_tuple)
        results = self.process_executor.map(getSceneMP, bo_tuple_list)
        
        for result in results:
            scene_tensor = result[0]
            masks = result[1] 
            if img_batch is None:
                img_batch = scene_tensor
            else:
                img_batch = torch.cat((img_batch, scene_tensor), 0)
            if mask_batch is None:
                mask_batch = [masks]
            else:
                mask_batch.append(masks)
        return img_batch, label_batch, objects_batch, mask_batch
        
        
        
def getSceneMP(args):
    scene = args[0] 
    object_images = args[1]  
    init_scaling_factor_range = args[2]
    scaling_factor = args[3]
    border = args[4]
    device = args[5]
                 
    channel, height, width = scene.shape
    masks = None
    init_scaling_factor = random.uniform(init_scaling_factor_range[0], init_scaling_factor_range[1])
    level = len(object_images)
    for object_img in object_images:
        scale = init_scaling_factor * (scaling_factor ** level)
        level -= 1
        _, _, h, w = object_img.shape
        #if max(h, w) > min(height, width) - self.border*2:
        #    scale *= (min(height, width) - self.border*2) / max(h, w)
        object_transform = transforms.Compose([
            transforms.Resize((int(scale*h), int(scale*w)))
        ])
        object_img = object_transform(object_img)
        object_img_t = object_img[0,:,:,:].to(device)
        _, hs, ws = object_img_t.shape
        #print(width, height)
        #print(ws, hs)
        destX = random.randint(border, width - ws - border)
        destY = random.randint(border, height - hs - border)
        mask_background = torch.zeros((channel, height, width)).to(device)
        #print(np_object_img.shape)
        mask_background[:, destY:destY+hs, destX:destX+ws] += object_img_t
        mask_background[mask_background > 0] = 1
        if masks is None:
            masks = [mask_background]
        else:
            masks.append(mask_background)
        scene_mask = torch.ones((channel, height, width)).to(device) - mask_background
        scene *= scene_mask
        #print(np_object_img.shape)
        scene[:, destY:destY+hs, destX:destX+ws] += object_img_t
    #scene = scene[np.newaxis, :, :, :]
    #scene = np.moveaxis(scene, 0, -1)
    #print(scene.shape)
    scene_tensor = torch.unsqueeze(scene, 0).to(device)
    final_transform = transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    scene_tensor = final_transform(scene_tensor)
    return (scene_tensor, masks)                 
                
                
        
        

