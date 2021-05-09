cd background_download
mkdir ../Dataset/background/
mogrify -path ../Dataset/background/ -format jpg *
python ../background_preprocess.py
