from PIL import Image
import os

for image_name in os.listdir('./Individual_masked/background/'):
    img = Image.open('./Individual_masked/background/'+image_name)
    try:
        img.verify()
    except Exception as e:
        print(e)
        print('Removing the file')
        os.remove('./Individual_masked/background/'+image_name)