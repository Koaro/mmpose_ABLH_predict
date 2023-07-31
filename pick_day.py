import json
import os
from tqdm import tqdm
import random
random.seed(50)


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


json_path = 'D:/data/annotations/128x128_train_33242.json'
pick_num = 10

with open(json_path, newline='') as f:
    source = json.load(f)

out10 = jsonbase()
out6 = jsonbase()
out3 = jsonbase()

current_m = int(source['images'][0]['file_name'][4:6])
anchor = 0 # 用來記錄上個月區間的起始位置
day_apear = [] # 用來記錄這個月出現的日
logs = []

for i in range(len(source['images'])):
    if i+1 >= len(source['images']) or current_m != int(source['images'][i]['file_name'][4:6]): # next month 或最後一筆資料 開始隨機選天
        
        try: # 開始隨機選天 10
            picked_10day = random.sample(day_apear, 10)
        except:
            picked_10day = day_apear
        try: # 開始隨機選天 6
            picked_6day = random.sample(picked_10day, 6)
        except:
            picked_6day = picked_10day
        try: # 開始隨機選天 3
            picked_3day = random.sample(picked_6day, 3)
        except:
            picked_3day = picked_6day

        picked_10day.sort()
        picked_6day.sort()
        picked_3day.sort()

        log_m = f"{current_m}: {picked_10day}, {picked_6day}, {picked_3day}"
        logs.append(log_m)
        print(log_m)

        for j in range(anchor, i): # 10
            d = int(source['images'][j]['file_name'][6:8])

            if d in picked_10day:
                out10['images'].append(source['images'][j])
                out10['annotations'].append(source['annotations'][j])

        for j in range(anchor, i): # 6
            d = int(source['images'][j]['file_name'][6:8])

            if d in picked_6day:
                out6['images'].append(source['images'][j])
                out6['annotations'].append(source['annotations'][j])

        for j in range(anchor, i): # 3
            d = int(source['images'][j]['file_name'][6:8])

            if d in picked_3day:
                out3['images'].append(source['images'][j])
                out3['annotations'].append(source['annotations'][j])

        
        current_m = int(source['images'][i]['file_name'][4:6])
        day_apear = []
        anchor = i
    
    # 紀錄每個月出現的天數
    current_d = int(source['images'][i]['file_name'][6:8])
    if current_d not in day_apear:
        if not(current_m == 7 and ( current_d >= 14 and current_d <= 18 )): # 不要 7/14~18
            day_apear.append(current_d)


out_path = 'D:/data/annotations/'
out_name = '128x128_train_33242'
with open(os.path.join(out_path, out_name+"_10day.json"), 'w') as file:
    json.dump(out10, file)

with open(os.path.join(out_path, out_name+"_6day.json"), 'w') as file:
    json.dump(out6, file)

with open(os.path.join(out_path, out_name+"_3day.json"), 'w') as file:
    json.dump(out3, file)

with open(f"{out_path}{out_name}_split_log.txt", 'w') as file:
    file.write('days picked in each month\n')
    for log in logs:
        file.write(f"{log}\n")


