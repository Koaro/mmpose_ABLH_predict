import os 
import json
import copy

import matplotlib.pyplot as plt

import random
random.seed(50)


if __name__ == "__main__":
    json_root = 'D:/data/annotations/'
    file_name = '128x128_train_33242_10day'
    json_path = os.path.join(json_root, f"{file_name}.json")


    with open(json_path, newline='') as file:
        json_file = json.load(file)

########## count ############ 
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

    count = []

    for l in range(1, int(max(keypoint))+1):
        count.append(keypoint.count(l))
#############################

    sorted = copy.deepcopy(count)
    sorted.sort()

    second_max = sorted[-2]
    max_cap = second_max * 10 # 最多的數量限制在次多的10倍

    most_level = count.index(sorted[-1])+1
    # print(most_level) # 確認最多是哪個level
    print(f"total: {len(keypoint)}")
    print(f"max: {sorted[-1]}, most level: {most_level}, cap: {max_cap}")

##### 找出所有level1的位置 #####
    level_index = []

    for i, key in enumerate(keypoint):
        if key == most_level:
            level_index.append(i)

    '''# testing
    test = []
    for i in level_index:
        test.append(json_file['annotations'][i]["keypoints"][1])
    print(len(test))
    plt.plot(list(range(len(test))), test)
    plt.show()'''

    cut = sorted[-1] - max_cap
    cut_data = random.sample( level_index, cut)
    cut_data.sort()
    print(f"cut: {len(cut_data)}")

###### cut ######

    offset = 0
    for cut_i in cut_data:
        del json_file['images'][cut_i-offset]
        del json_file['annotations'][cut_i-offset]
        offset+=1

    '''# testing
    print(f"{len(json_file['annotations'])}, {len(json_file['images'])}")
    
    keypoint = []
    for js in json_file["annotations"]:
        anno_keypoint = js["keypoints"][1]

        if interpolated :
            anno_keypoint/=2

        keypoint.append(anno_keypoint)

    count = []
    for l in range(1, int(max(keypoint))+1):
        count.append(keypoint.count(l))

    for i, c in enumerate(count):
        print(f"level {i+1}: {c} | {'{:.2f}'.format(c/len(json_file['images'])*100)}%")

    plt.bar(list(range(len(count))), count)
    plt.show()'''

##### output #####
    out_root = 'D:/data/annotations/clamp10x/'

    if not os.path.exists(out_root):
        os.mkdir(out_root)

    out_path = os.path.join(out_root, f"{file_name}_clamp10x.json")

    with open(out_path, 'w') as file:
        json.dump(json_file, file)
