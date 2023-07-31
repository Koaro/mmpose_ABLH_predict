import csv 
import json
import os
from tqdm import tqdm

out_list = []

# generate from raw & json
csv_root = 'D:/LIDAR/gt/all.csv'
json_path = 'D:/data/annotations/128x128_test_record.json'

gt_list = []

with open(csv_root, newline='') as file:
    rows = csv.reader(file)

    for i, row in enumerate(rows):
        date = row[0].strip()
        height = row[1].strip()

        if height != 'NaN' and len(height) > 0:
            try:
                level = int(round(((int(height)-51)/26), 0))+1
            except:
                print(f"[{i}]date: [{date}], height: [{height}]")
            gt_list.append([date, height, level])

        # if i == 26976:
        #     print(f"[{i}]date: [{date}], height: [{height}]")

with open(json_path, newline='') as file:
    jfile = json.load(file)
    
    for i in range(len(jfile['annotations'])):
        img_id = str(jfile['annotations'][i]['image_id'])
        date = f'{img_id[:4]}/{int(img_id[4:6])}/{int(img_id[6:8])} {img_id[8:10]}:{img_id[10:12]}'

        found = False
        for j in range(len(gt_list)):
            if gt_list[j][0] == date:
                out_list.append(gt_list[j])
                found = True
                break
        
        if not found:
            print(f"{date} not found.")


'''# generate from json
# json_path = 'D:/data/annotations/128pad256/128pad256_train_new_clamped_33242_6day.json'
json_path = 'D:/data/annotations/128pad256/128pad256_val_clamped_33242.json'
with open(json_path, newline='') as file:
    jfile = json.load(file)

    for i in range(len(jfile['annotations'])):
        level = jfile['annotations'][i]['keypoints'][1]
        height = (level-1) * 26 + 51

        img_id = str(jfile['annotations'][i]['image_id'])
        date = f'{img_id[:4]}/{img_id[4:6]}/{img_id[6:8]} {img_id[8:10]}:{img_id[10:12]}'

        out_list.append([date, height, level])
'''

'''# generate from csv
# csv_path = 'D:/data/csv_without_vertical/'
csv_path = '../test_data/csv/'
csv_names = os.listdir(csv_path)
for csv_n in tqdm(csv_names, desc='read csv'):
    with open(csv_path+csv_n, newline='') as csv_f:
        next(csv_f)
        row = next(csv_f).split(',')
        date = f"{row[0][:4]}/{row[0][4:6]}/{row[0][6:14]}"
        height = row[-1].strip()

        level = int(round(((int(height)-51)/26), 0))+1

        out_list.append([date, height, level])
'''

# output
out_csv_path = './gt_list_test_record.csv'
with open(out_csv_path, 'w', newline='') as out_file:
    writer = csv.writer(out_file)

    writer.writerow(['date', 'pblh', 'gt'])

    for o in tqdm(out_list, desc='write csv'):
        writer.writerow(o)

