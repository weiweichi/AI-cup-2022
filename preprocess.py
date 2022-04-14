
import glob, os
from args import args

root = './data'
tot = 0

for idx, name in enumerate(args['dir_name']):
    dir_path = os.path.join(root, name)
    print(f"open {dir_path} folder, renaming...")

    for i, file in enumerate( glob.glob( os.path.join(dir_path, "*"))):
        new_file = os.path.join(dir_path, f"{idx}_{i}.jpg")
        os.rename(file, new_file)

    print(f"finishing! number of images: {i+1}")
    tot += i+1

print("------------------------------")
print(f"number of total sample: {tot}")
# number of total sample:80394