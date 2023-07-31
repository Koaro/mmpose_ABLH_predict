import json
from tqdm import tqdm
import os

def jsonbase():
    info = {
        "description": "COCO 2017 Dataset",
        "url": "http://cocodataset.org",
        "version": "1.0",
        "year": 2017,
        "contributor": "COCO Consortium",
        "date_created": "2017/09/01"
    }

    data = {'info':info, 'licenses':[], 'images':[], 'annotations':[], 'categories':[]}

    data["licenses"].append(
        {
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License"
        }
    )

    data['categories'].append({
            "supercategory": "person",
            "id": 1,
            "name": "person",
            "keypoints": [
                "nose"
            ],
            "skeleton": []
        })

    return data

# json_path = 'D:/data/annotations/all_clamped_33242.json'
# json_root = 'D:/data/annotations/'
# json_root = '../test_data/annotation/'
json_root = 'D:/data/annotations/tmp/'

'''# 只改一個json
json_name = '128X128_all_33242.json'
# json_name = 'train_LIDAR_10day.json'
json_path = os.path.join(json_root, json_name)

with open(json_path, newline='') as file:
    json_file = json.load(file)

#######################################################################
for j in tqdm(range(len(json_file['images'])), desc=json_name):
    # adjust height
    json_file['images'][j]["width"] = 184
    json_file['images'][j]["height"] = 256
    json_file['annotations'][j]["area"] = 184*256
    json_file['annotations'][j]["bbox"] = [0,0,184,256]

    # gt 
    if len(json_file['annotations'][j]['keypoints']) > 0:
        json_file['annotations'][j]['keypoints'][1] *= 2
        json_file['annotations'][j]['keypoints'][0] = 183/2

#######################################################################
out_root = 'D:/data/annotations/'
# out_root = json_root
out_name = '184x256_all_33242.json'
out_path = os.path.join(out_root, out_name)

with open(out_path, 'w') as file:
    json.dump(json_file, file)'''


# 修改整個資料夾的json
file_list = os.listdir(json_root)
for i in range(len(file_list)):
    if file_list[i][-4:] == 'json':
        json_path = os.path.join(json_root, file_list[i])
    else:
        continue

    with open(json_path, newline='') as file:
        json_file = json.load(file)

    #######################################################################
    for j in tqdm(range(len(json_file['images'])), desc=file_list[i]):
        # adjust height
        json_file['images'][j]["width"] = 184
        json_file['images'][j]["height"] = 192
        json_file['annotations'][j]["area"] = 184*192
        json_file['annotations'][j]["bbox"] = [0,0,184,192]

        # gt *
        if len(json_file['annotations'][j]['keypoints']) > 0:
            json_file['annotations'][j]['keypoints'][1] *= 2
            json_file['annotations'][j]['keypoints'][0] = 183/2

    #######################################################################
    out_root = 'D:/data/annotations/184x192_interpolated/'

    if not os.path.exists(out_root):
        os.mkdir(out_root)

    out_path = os.path.join(out_root, f"184x192_inter_{file_list[i][8:]}")
    # out_path = json_path

    with open(out_path, 'w') as file:
        json.dump(json_file, file)
