import json
import os
from tqdm import tqdm

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

json_path = 'D:/data/annotations/all_clamped_33242.json'

with open(json_path, newline='') as f:
    source = json.load(f)

total = len(source["images"])
test_num = 3000
val_num = 3000

title = ['train', 'val', 'test']
task_amount = [total-test_num-val_num, val_num, test_num]

out = jsonbase()
out_path = 'D:/data/annotations/'
out_name = '_clamped_33242.json'

offset = 0
for task in range(3):
    for i in tqdm(range(task_amount[task]), desc=title[task]):
        out['images'].append(source['images'][offset+i])
        out['annotations'].append(source['annotations'][offset+i])

    offset += task_amount[task]
    
    with open(os.path.join(out_path, title[task] + out_name), 'w') as file:
        json.dump(out, file)
    out = jsonbase()



