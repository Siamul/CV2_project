import os
import torch
import cv2
from clutterized_dataloader import clutterized_datasetLoader
import torchvision
import time
import sys
import gc

cluttered_dataloader = clutterized_datasetLoader(dataset_location = './Dataset/', device_str="cuda", batch_size=32)

averageAcross = 100
sys.stdout = open('dataloader_performance_comparison.txt', 'w')

gpu_time = 0
for i in range(averageAcross):
    start = time.process_time()
    cluttered_dataloader.getBatch()
    gpu_time += (time.process_time() - start)

print("Time taken for processing using GPU: ", gpu_time/averageAcross)

del cluttered_dataloader
gc.collect()

cluttered_dataloader2 = clutterized_datasetLoader(dataset_location = './Dataset/', device_str="cpu", batch_size=32)

sqcpu_time = 0
for i in range(averageAcross):
    start = time.process_time()
    cluttered_dataloader2.getBatch(multi_process=False)
    sqcpu_time += (time.process_time() - start)

print("Time taken for processing sequentially using CPU: ", sqcpu_time/averageAcross)

del cluttered_dataloader2
gc.collect()

cluttered_dataloader3 = clutterized_datasetLoader(dataset_location = './Dataset/', device_str="cpu", batch_size=32)

mpcpu_time = 0
for i in range(averageAcross):
    start = time.process_time()
    cluttered_dataloader3.getBatch(multi_process=True)
    mpcpu_time += (time.process_time() - start)

print("Time taken for processing parallely using CPU: ", mpcpu_time/averageAcross)
sys.stdout.close()