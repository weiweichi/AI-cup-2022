
import glob, os
from args import args

# we store our zip in download
if not os.path.isdir('download'):
    os.mkdir('download')
    # your link
    link = "http://aicup-dataset.aidea-web.tw:18080/dataset/train/banana.zip \
        http://aicup-dataset.aidea-web.tw:18080/dataset/train/bareland.zip \
        http://aicup-dataset.aidea-web.tw:18080/dataset/train/carrot.zip \
        http://aicup-dataset.aidea-web.tw:18080/dataset/train/corn.zip \
        http://aicup-dataset.aidea-web.tw:18080/dataset/train/dragonfruit.zip \
        http://aicup-dataset.aidea-web.tw:18080/dataset/train/garlic.zip \
        http://aicup-dataset.aidea-web.tw:18080/dataset/train/guava.zip \
        http://aicup-dataset.aidea-web.tw:18080/dataset/train/inundated.zip \
        http://aicup-dataset.aidea-web.tw:18080/dataset/train/peanut.zip \
        http://aicup-dataset.aidea-web.tw:18080/dataset/train/pineapple.zip \
        http://aicup-dataset.aidea-web.tw:18080/dataset/train/pumpkin.zip \
        http://aicup-dataset.aidea-web.tw:18080/dataset/train/rice.zip \
        http://aicup-dataset.aidea-web.tw:18080/dataset/train/soybean.zip \
        http://aicup-dataset.aidea-web.tw:18080/dataset/train/sugarcane.zip \
        http://aicup-dataset.aidea-web.tw:18080/dataset/train/tomato.zip" 
    os.system(f'wget -t0 -c -p ./download {link}')

root = 'data'
# unzip all zips in ./data folder
if not os.path.isdir(root):
    os.system('unzip "./download/*.zip" -d ./data')

# rename data as {label_id}_{file_id}.jpg
tot = 0
for idx, name in args['label2name'].items():
    dir_path = os.path.join(root, name)
    print(f"open {dir_path} folder, renaming...")

    ls_all_data = sorted(glob.glob( os.path.join(dir_path, "*")))
    for i, file in enumerate(ls_all_data):
        new_file = os.path.join(dir_path, f"{idx}_{i}.jpg")
        os.rename(file, new_file)

    print(f"finishing! number of {name} images: {len(ls_all_data)}\n")
    tot += len(ls_all_data)

print(f"number of total images: {tot}")
# number of total images:80394