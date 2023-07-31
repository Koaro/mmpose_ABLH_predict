import os
import json
import csv

import numpy as np
import matplotlib.pyplot as plt

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

def level_count(json_file):
    keypoint = []
    id = []

    if json_file["annotations"][0]['bbox'][3] == 256:
        interpolated = True
    else:
        interpolated = False

    for js in json_file["annotations"]:
        anno_keypoint = js["keypoints"][1]

        if interpolated :
            anno_keypoint/=2

        keypoint.append(anno_keypoint)
        id.append(js["image_id"])

    # print(keypoint.count(1))
    level = []

    for l in range(1, int(max(keypoint))+1):
        level.append(keypoint.count(l))

    for i, count in enumerate(level):
        print(f"level {i+1}: {count} | {'{:.2f}'.format(count/len(json_file['images'])*100)}%")

def varify_gt(json_file):
    csv_root = 'D:/LIDAR/gt/all.csv'
    
    gt = {'id':[], 'gt':[]}
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
                
                ymd, time = date.split()
                y,m,d = ymd.split("/")
                m = m.zfill(2)
                d = d.zfill(2)

                h,M = time.split(":")

                date_norm = f"{y}{m}{d}{h}{M}"

                gt['id'].append(int(date_norm))
                gt['gt'].append(level)

            # if i == 26976:
            #     print(f"[{i}] date: [{date}], height: [{height}]")
            #     print(f"ymd: {ymd}, time: {time}")
            #     print(f"{y}|{m}|{d}||{h}|{M}")

    if json_file["annotations"][0]['bbox'][3] == 256:
        interpolated = True
    else:
        interpolated = False

    error_list = {'j_id':[], 'j_gt':[], 'g_id':[], 'g_gt':[]}
    for js in tqdm(json_file["annotations"], desc='checking'):
        index = gt['id'].index(js['image_id'])
        anno_keypoint = js["keypoints"][1]

        if interpolated:
            anno_keypoint /= 2

        if anno_keypoint != gt["gt"][index]:
            error_list["j_id"].append(js["image_id"])
            error_list["j_gt"].append(js["keypoints"][1])

            error_list["g_id"].append(gt['id'][index])
            error_list["g_gt"].append(gt["gt"][index])

    if len(error_list["j_id"]) == 0:
        print("gt no error.")
    else:
        print("there's error in gt, check error log.")
        
        current_path = os.path.dirname(os.path.abspath(__file__))
        with open(f"{current_path}/error_log.txt", 'w') as log:
            for i in range(len(error_list["j_id"])):
                row = f'annotation: [{error_list["j_id"][i]}, {error_list["j_gt"][i]}], gt: [{error_list["g_id"][i]}, {error_list["g_gt"][i]}]' 

                log.write(f"{row}\n")

def varify_day(json_file):

    m_list = [[],[],[],[],[],[],[],[],[],[],[],[]]

    for js in json_file['images']:
        m = int(js['file_name'][4:6])
        d = int(js['file_name'][6:8])
        
        if d not in m_list[m-1]:
            m_list[m-1].append(d)

    for i in range(12):
        print(f"days in {i+1}: {m_list[i]}")

def varify_data(json_file):
    prob_list = []
    # check formation
    ### check image & annotation
    image_anno_fine = True
    for i in range(len(json_file['images'])):
        # check match
        if json_file['images'][i]["id"] != json_file['annotations'][i]["image_id"]:
            image_anno_fine = False
            prob = {"file":json_file['images'][i]['file_name'], "desc":"image & anno not match."}
            prob_list.append(prob)

        if str(json_file["images"][i]["id"]) !=  json_file["images"][i]['file_name'][:-4]:
            image_anno_fine = False
            prob = {"file":json_file['images'][i]['file_name'], "desc":"file_name & id not match."}


    print(f"image & anno: {image_anno_fine}")
    # check GT
    pass

def plot_gt(json_file):
    gt_list = []

    if json_file["annotations"][0]['bbox'][3] == 256:
        interpolated = True
    else:
        interpolated = False

    for js in json_file['annotations']:
        anno_keypoint = js["keypoints"][1]

        if interpolated :
            anno_keypoint/=2

        gt_list.append(anno_keypoint)

    fig, axes = plt.subplots(2, 1)
    fig.set_figheight(8)
    fig.set_figwidth(10)

    axes1 = axes[0]

    axes1.set_title("gt")
    axes1.set_ylabel("level")
    axes1.set_yticks(list(range(0, int(max(gt_list)), 10)))
    axes1.set_yticks(list(range(int(max(gt_list)))), minor=True)

    axes1.plot(list(range(len(json_file['annotations']))), gt_list, linewidth=1)
    axes1.grid(axis='y')
    axes1.grid(which='minor', alpha=0.3)
    # plt.show()

    ######################

    keypoint = []
    for js in json_file["annotations"]:
        anno_keypoint = js["keypoints"][1]

        if interpolated :
            anno_keypoint/=2

        keypoint.append(anno_keypoint)

    # print(keypoint.count(1))
    level = []

    for l in range(1, int(max(keypoint))+1):
        level.append(keypoint.count(l))

    axes2 = axes[1]
    axes2.set_title("gt level count")
    axes2.set_ylabel("count")
    axes2.set_yticks(range(0, max(level)+100, 100), minor=True)
    axes2.set_xlabel("level")
    axes2.set_xticks(list(range(0, len(level)+1, 5)))
    axes2.set_xticks(list(range(len(level)+1)), minor=True)

    axes2.bar(list(range(1,len(level)+1)), level)
    axes2.grid()
    axes2.grid(which='minor', alpha=0.3)

    plt.show()


def plot_gt_by_count(json_file):
    keypoint = []

    if json_file["annotations"][0]['bbox'][3] == 256:
        interpolated = True
    else:
        interpolated = False

    for js in json_file["annotations"]:
        anno_keypoint = js["keypoints"][1]

        if interpolated :
            anno_keypoint/=2

        keypoint.append(anno_keypoint)

    # print(keypoint.count(1))
    level = []

    for l in range(1, int(max(keypoint))+1):
        level.append(keypoint.count(l))


    fig, axes = plt.subplots()

    axes.set_yticks(range(0, max(level)+100, 100), minor=True)
    axes.set_xticks(list(range(len(level))), minor=True)


    plt.bar(list(range(1,len(level)+1)), level)
    plt.grid()
    plt.grid(which='minor', alpha=0.3)
    plt.show()


if __name__ == "__main__":
    json_path = 'D:/data/annotations/128x128_train_record_3day.json'
    # json_path = 'D:/data/annotations/128x128_all_33242.json'
    # json_path = 'D:/data/annotations/128x128_val_33242.json'
    # json_path = 'D:/data/annotations/clamp10x/128x128_train_33242_3day_clamp10x.json'
    # json_path = 'D:/data/annotations/record/train_LIDAR_6day.json'
    # json_path = 'D:/data/annotations/record/val_LIDAR.json'
    # json_path = 'D:/data/annotations/record/test_LIDAR.json'
    # json_path = 'D:/data/annotations/184x192_interpolated/184x192_inter_val_record.json'


    with open(json_path, newline='') as file:
        json_file = json.load(file)

    print(f"{json_path.split('/')[-1]}")
    print(f"data count: {len(json_file['images'])}")
    # level_count(json_file)
    # varify_day(json_file)
    # varify_data(json_file)
    # varify_gt(json_file)
    plot_gt(json_file)
    # plot_gt_by_count(json_file)


    print("")

    # json_path = 'D:/data/annotations/128X128/128X128_train_new_clamped_33242_6day.json'
    json_path = 'D:/data/annotations/128x128_train_record_6day.json'
    with open(json_path, newline='') as file:
        json_file = json.load(file)
    print(f"{json_path.split('/')[-1]}")
    print(f"data count: {len(json_file['images'])}")
    # varify_day(json_file)
    plot_gt(json_file)

    print("")

    # json_path = 'D:/data/annotations/128X128/128X128_train_new_clamped_33242_3day.json'
    json_path = 'D:/data/annotations/128x128_train_record_10day.json'
    with open(json_path, newline='') as file:
        json_file = json.load(file)
    print(f"{json_path.split('/')[-1]}")
    print(f"data count: {len(json_file['images'])}")
    # varify_day(json_file)
    plot_gt(json_file)

    # print("")

    # json_path = 'D:/data/annotations/128X128/128X128_all_33242_LIDAR.json'
    # with open(json_path, newline='') as file:
    #     json_file = json.load(file)
    # print(f"{json_path.split('/')[-1]}")
    # print(f"data count: {len(json_file['images'])}")
    # level_count(json_file)
    # # varify_day(json_file)