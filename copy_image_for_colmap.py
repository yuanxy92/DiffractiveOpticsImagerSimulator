import os 
import os.path
import shutil

root_folder = 'C:/Projects/Code/Aurora_papers/data/arm_3cam'
src_folder = f'{root_folder}/data_0823_6_corrected'
dst_folder = f'{root_folder}/data_0823_6_corrected_colmap/images'

os.makedirs(dst_folder, exist_ok=True)

for idx in range(1, 2000, 5):
    for cam_idx in range(1, 4):
        filename = f'camera_{cam_idx}_frame_{idx}_corrected.png'
        srcname = f'{src_folder}/{filename}'
        dstname = f'{dst_folder}/{filename}'
        if os.path.isfile(srcname):
            shutil.copyfile(srcname, dstname)


