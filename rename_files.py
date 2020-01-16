import os
dir_path = '/cs/labs/josko/aszeskin/Rafi_Tumor_data/allFU_liverSeg'
for filename in os.listdir(dir_path):
    new_filename = filename.replace('allFU_', 'FU')
    new_filename = new_filename.replace('liverSeg', 'liverseg')
    old_file_path = os.path.join(dir_path, filename)
    new_file_path = os.path.join(dir_path, new_filename)
    os.rename(old_file_path, new_file_path)
    print(old_file_path, new_file_path)