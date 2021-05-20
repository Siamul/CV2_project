# CSE 60536 Computer Vision II 
## Semester Project
This repository contains the data and codes for the semester project. 
The midterm project report (contains both Charlie's and my results) can be found here: https://docs.google.com/document/d/1_aEPQlDVgM4MKWv7bmCKboZAt88hgzc8Ua5nBMiE4wM/
The final project report (contains both Charlie's and my results) can be found here: https://docs.google.com/document/d/1U4k5R0X0ijcY43Fd0yVzeveOgfMTpCDuaeY8Ao72bVU/

Instructions on how to use the Dataloader:

Use the implementation on masked_clutterized_dataloader.py as this works directly with labelme annotated images. The initial implementation uses cropped out images to create the cluttered object images however, there is a problem in using this approach if the color used to fill the cropped out background matches the color of the object (my mug was black and some of the pixels inside the cup where getting replaced by the background pixel).

