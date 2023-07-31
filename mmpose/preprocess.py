# 把lidar資料轉成每15分鐘的資料，並做成圖片

import csv
import os
import numpy as np
from tqdm import tqdm
import json

from PIL import Image
import math

import time


isPreData = False
isAbuse = False

testrun = False

class Config():
    def __init__(self, 
                 data_root, 
                 out_img_path, 
                 out_annotations_path, 
                 gt_path = '', 
                 split=[], 
                 feature=[], 
                 makeimg=True, 
                 clamp=False, 
                 makeJson=True,
                 interpolation=False):

        self.data_root = data_root
        self.out_img_path = out_img_path
        self.out_annotations_path = out_annotations_path
        self.gt_path = gt_path
        self.split = split
        self.feature = feature
        self.makeimg = makeimg
        self.clamp = clamp
        self.makeJson = makeJson
        self.interpolation = interpolation
        


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

def indexOfAPeriod(_period, _rows): # 找到當前區間的起點index
    hour = int(_period[0:2])
    min = int(_period[2:4])

    for index in range(0, len(_rows)-1):
        if _period[2:4] == '00' and int(_rows[index][0].split()[1][0:2]) < hour and int(_rows[index+1][0].split()[1][0:2]) == hour:
            return index
        else:
            if _rows[index][0].split()[1][0:2] == _period[0:2] and int(_rows[index][0].split()[1][3:5]) < min and int(_rows[index+1][0].split()[1][3:5]) == min:
                return index
    return -1

def isCsvCompleteHead(_rows): # 光達資料從00:00開始
    if _rows[0][0].split()[1][0:5] == '00:00':
        return True
    else:
        return False
def isCsvCompleteTail(_rows): # 光達資料在23:59結束 # 沒用到
    if _rows[len(_rows)-1][0].split()[1][0:5] == '23:59':
        return True
    else:
        return False

def nextPeriod(_period): 
    if _period[2:4] == '45':
        return str(int(_period[0:2]) + 1).zfill(2) + '00'
    else:
        return _period[0:2] + str(int(_period[2:4]) + 15)
def isClosest(_time1, _time2):
    if 60 - float(_time1[_time1.find(':', 3)+1:]) < float(_time2[_time2.find(':', 3)+1:]):
        return True
    else:
        return False

def myOwnRound(num): # 自製四捨五入
    if int(str(num)[str(num).find('.')+1]) >= 5:
        return int(num) + 1
    else:
        return int(num)

def MakeImg( dataList:list, period:str, ground, cfg ):
    # print(dataList[0])
    ############# if img exist #############
    if not os.path.exists( cfg.out_img_path ):
        os.makedirs( cfg.out_img_path )

    img_name = f"{dataList[0]['time'][:8]}{period}"

    if os.path.exists(f"{cfg.out_img_path}/{img_name}.png"): # 如果存在同名的img就不會產生新的img
        # print(f"img:{img_name} exist.")
        return img_name

    ############# normalize #################
    t_nor_s = time.time()

    afterNormalizeR = []
    afterNormalizeG = []
    afterNormalizeB = []
    tempR = 0
    tempG = 0
    tempB = 0
        
    afterNormalizeT = []
    afterNormalizeH = []
    afterNormalizeP = []
    tempT = 0
    tempH = 0
    tempP = 0

    for i, data in enumerate(dataList):
        r = float(data['rw'])
        g = float(data['snr'])
        b = float(data['si'])

        if r >= 0:
            tempR = int(round((127+((r/0.836)**(2/3))*10), 0))
            
            if tempR < 0:
                tempR = 0
            elif tempR > 255:
                tempR = 255 
        else:
            tempR = int(round((127-(((-r/0.836))**(2/3))*10), 0))
            if tempR < 0:
                tempR = 0
            elif tempR > 255:
                tempR = 255
        afterNormalizeR.append(tempR)
        
        if g == 0.0:
            tempG = 0
        else:
            tempG = int(round(((math.log(g,10)+3)*32), 0))
        if tempG < 0:
            tempG = 0
        elif tempG > 255:
            tempG = 255
        afterNormalizeG.append(tempG)

        if b >= 0:
            tempB = int(round(((math.log(b+1, 4)+1)*32), 0))
            if tempB < 0:
                tempB = 0
            elif tempB > 255:
                tempB = 255
        else:
            tempB = int(round(((1-(math.sqrt(-b)))*32), 0))
            if tempB < 0:
                tempB = 0
            elif tempB > 255:
                tempB = 255
        afterNormalizeB.append(tempB) 

        if i % 192 == 0:
            t = float(data['temp'])
            h = float(data['humi'])
            p = float(data['press'])

            if t > 35.5:
                tempT = 255
            elif t < 10:
                tempT = 0
            else:
                tempT = round((t-10)*10, 0)
            afterNormalizeT.append(tempT) 

            if h > 100: 
                tempH = 255
            elif h < 49:
                tempH = 0
            else:
                tempH = round((h-49)*5, 0)
            afterNormalizeH.append(tempH)

            if p > 1021:
                tempP = 255
            elif p < 970:
                tempP = 0
            else:
                tempP = round((p-970)*5, 0)
            afterNormalizeP.append(tempP) 

    t_nor_e = time.time()
    ################## make picture ###################
    x = int(len(dataList)/192)		#width 時間(15min)    # len(dataList) = 35328 = 184 * 192 
    y = 256 if cfg.interpolation else 192  #height 高度 ######################################################################

    # feature handle
    rw = 'rw' in  cfg.feature # R
    snr = 'snr' in  cfg.feature # G
    si = 'si' in  cfg.feature # B

    if not rw and not snr and not si: # 無指定 == 3 feature
        rw = True
        snr = True
        si = True

    img_array = []
    t_img_s = time.time()
    
    if ground == '0': # 有無地表資料
        if cfg.interpolation:
            for i in range(0, x):
                index = 0
                line = []
                for j in range(0, y): # [0] [0+1/2] [1] [1+2/2] [2] ...
                    if j % 2 == 0:
                        red = int(afterNormalizeR[j-index+i*192])
                        grn = int(afterNormalizeG[j-index+i*192])
                        blu = int(afterNormalizeB[j-index+i*192])
                        # line.append([int(afterNormalizeR[j-index+i*192]), int(afterNormalizeG[j-index+i*192]), int(afterNormalizeB[j-index+i*192])])
                        index += 1
                    else:
                        red = (int(afterNormalizeR[j-index+i*192]) + int(afterNormalizeR[j-index+1+i*192])) / 2
                        grn = (int(afterNormalizeG[j-index+i*192]) + int(afterNormalizeG[j-index+1+i*192])) / 2
                        blu = (int(afterNormalizeB[j-index+i*192]) + int(afterNormalizeB[j-index+1+i*192])) / 2

                    # feature handle
                    t_r = myOwnRound(red) if rw else 0
                    t_g = myOwnRound(grn) if snr else 0
                    t_b = myOwnRound(blu) if si else 0
                    pixel = [t_r, t_g, t_b]

                    if pixel.count(0) == 2: # 1 feature
                        non_zero = [x for x in pixel if x != 0]
                        t_r = non_zero[0]
                        t_g = non_zero[0]
                        t_b = non_zero[0]
                        pixel = [t_r, t_g, t_b]

                    line.append(pixel)

                img_array.append(line)
        else:
            for i in range(0, x):
                index = 0
                line = []
                for j in range(0, y): # [0] [0+1/2] [1] [1+2/2] [2] ...
                    red = int(afterNormalizeR[j+i*192])
                    grn = int(afterNormalizeG[j+i*192])
                    blu = int(afterNormalizeB[j+i*192])

                    # feature handle
                    t_r = myOwnRound(red) if rw else 0
                    t_g = myOwnRound(grn) if snr else 0
                    t_b = myOwnRound(blu) if si else 0
                    pixel = [t_r, t_g, t_b]

                    if pixel.count(0) == 2: # 1 feature
                        non_zero = [x for x in pixel if x != 0]
                        t_r = non_zero[0]
                        t_g = non_zero[0]
                        t_b = non_zero[0]
                        pixel = [t_r, t_g, t_b]

                    line.append(pixel)

                img_array.append(line)
    elif ground == '1':
        if cfg.interpolation:
            for i in range(0, x):
                index = 2
                line = []
                for j in range(0, y):
                    pixel = []

                    if j == 0:
                        pixel = [int(afterNormalizeT[i]), int(afterNormalizeT[i]), int(afterNormalizeP[i])]
                    elif j == 1:
                        pixel = [0, 0, 0]
                    else:
                        if j % 2 == 0:
                            red = int(afterNormalizeR[j-index+i*192])
                            grn = int(afterNormalizeG[j-index+i*192])
                            blu = int(afterNormalizeB[j-index+i*192])
                            # line.append([int(afterNormalizeR[j-index+i*192]), int(afterNormalizeG[j-index+i*192]), int(afterNormalizeB[j-index+i*192])])
                            index += 1
                        else:
                            red = (int(afterNormalizeR[j-index+i*192]) + int(afterNormalizeR[j-index+1+i*192])) / 2
                            grn = (int(afterNormalizeG[j-index+i*192]) + int(afterNormalizeG[j-index+1+i*192])) / 2
                            blu = (int(afterNormalizeB[j-index+i*192]) + int(afterNormalizeB[j-index+1+i*192])) / 2

                        # feature handle
                        t_r = myOwnRound(red) if rw else 0
                        t_g = myOwnRound(grn) if snr else 0
                        t_b = myOwnRound(blu) if si else 0
                        pixel = [t_r, t_g, t_b]

                        if pixel.count(0) == 2: # 1 feature
                            non_zero = [x for x in pixel if x != 0]
                            t_r = non_zero[0]
                            t_g = non_zero[0]
                            t_b = non_zero[0]
                            pixel = [t_r, t_g, t_b]

                    line.append(pixel)

                img_array.append(line)
        else:
            for i in range(0, x):
                index = 2
                line = []
                for j in range(0, y):
                    pixel = []

                    if j == 0:
                        pixel = [int(afterNormalizeT[i]), int(afterNormalizeT[i]), int(afterNormalizeP[i])]
                    else:
                        red = int(afterNormalizeR[j-1+i*192])
                        grn = int(afterNormalizeG[j-1+i*192])
                        blu = int(afterNormalizeB[j-1+i*192])

                        # feature handle
                        t_r = myOwnRound(red) if rw else 0
                        t_g = myOwnRound(grn) if snr else 0
                        t_b = myOwnRound(blu) if si else 0
                        pixel = [t_r, t_g, t_b]

                        if pixel.count(0) == 2: # 1 feature
                            non_zero = [x for x in pixel if x != 0]
                            t_r = non_zero[0]
                            t_g = non_zero[0]
                            t_b = non_zero[0]
                            pixel = [t_r, t_g, t_b]

                    line.append(pixel)

                img_array.append(line)

    img_array = np.array(img_array)
    img_array = np.transpose(img_array, (1,0,2))
    img = Image.fromarray(np.uint8(img_array))

    t_img_e = time.time()
    if testrun: print(f'\n{img_name} | nor: {t_nor_e-t_nor_s} | img: {t_img_e-t_img_s}')

    img.save(f"{cfg.out_img_path}/{img_name}.png")
    return img_name

def MakeJson( cfg, gtList=[] ): # split = [test_num, val_num, train_num] 或是指定val和test 剩下當train
    idCount = 0
    
    imgDir = os.listdir(cfg.out_img_path)
    imgDir.sort()

    imgs = []
    for file_name in imgDir:
        imgs.append(str(file_name))

    data = jsonbase()

    if cfg.split: # 如果要切train, val, test
        task_num = 3
        des = ['train', 'val', 'test']

        if len(cfg.split) == 3: # test, val, train
            task = [cfg.split[2], cfg.split[1], cfg.split[0]]
        elif len(cfg.split) == 2: # test, val
            task = [len(imgs)-cfg.split[0]-cfg.split[1], cfg.split[1], cfg.split[0] ] # train = all-val-test
        elif len(cfg.split) == 1: # test
            task = [len(imgs)-cfg.split[0]-cfg.split[0], cfg.split[0], cfg.split[0] ] # train = all-val-test, val = test

    else:
        task_num = 1
        des = ['annotations']
        task = [len(imgs)]

    # clamp level1
    level1_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    level1_limit_1 = [56, 78, 114, 105, 76, 65, 81, 93, 78, 79, 110, 72] # 1成level1的數量
    limit_percent = 5 # 用來決定留幾成的level1

    level1_limit = [num * limit_percent for num in level1_limit_1]

    sample_img = Image.open(f'{cfg.out_img_path}{imgs[0]}')
    width, height = sample_img.size

    # width = 128
    # height = 256 if cfg.interpolation else 128

    offset = 0 # 幫助定址圖片
    for t in range(task_num):
        for i in tqdm(range(task[t]), desc=des[t]):
            # print(i)
            
            dict_ = []
            key = []

            if gtList: # have GT
                gt = 0
                for n in gtList:
                    if imgs[offset+i][:-4] == n['name']:
                        gt = n['gt']
                        break

                # 如果當月的level1已經達到上限，之後的level1資料將不寫入annotation，val和test不會限制level1數量
                if cfg.clamp and t == 1 and gt == 51 :
                    month_index = int(imgs[offset+i][4:6])-1
                    if level1_count[month_index] >= level1_limit[month_index]:
                        continue
                    else:
                        level1_count[month_index]+=1

                pixleGT = (int(round(((int(gt)-51)/26), 0))+1)
                if cfg.interpolation:
                    pixleGT *= 2

                key.append((width-1)/2)
                key.append(pixleGT)
                key.append(2)
            
            dict_ = { "segmentation": [],"num_keypoints": 1,"area": height*width,"iscrowd": 0,"keypoints": key,"image_id": int(imgs[offset+i][:-4]),"bbox": [0,0,width,height],"category_id": 1,"id": 999999-idCount }

            data['annotations'].append(dict_)

            #儲存images字段  
            dict_ = []
            dict_ = { "license": 1, "file_name": "", "coco_url": "", "height": height, "width": width, "date_captured": "", "flickr_url": "", "id": 1 }
            dict_['id'] = int(imgs[offset+i][:-4]) # id == file_name - 副檔名
            dict_['file_name'] = imgs[offset+i][:-4]+'.png'

            data['images'].append(dict_)

            key = []
            idCount += 1
    
        offset += task[t]
        if not os.path.exists(cfg.out_annotations_path):
            os.makedirs(cfg.out_annotations_path)

        isclamp = "_clamped"  if cfg.clamp else ""
        interpolated = 'interpolated_' if cfg.interpolation else ""
        title = 'annotations' if len(des) == 1 else des[t]

        json_name = f'{interpolated}{title}{isclamp}_33242.json'

        # if len(des) == 1:
        #     json_name = f'annotations{isclamp}.json'
        # else:
        #     json_name = f'{des[t]}{isclamp}_33242.json'

        file_n = os.path.join(cfg.out_annotations_path, json_name)
        with open( file_n, 'w' ) as obj:
            json.dump(data, obj)
        data = jsonbase() # clear data

count_img = 0 # debug 用
def ProcessRaw(data_root, out_img_path, out_annotations_path, gt_path = '', split=[], feature=[], makeimg=True, clamp=False, makeJson=True, interpolation=True ):
    # make config
    cfg = Config(data_root, out_img_path, out_annotations_path, gt_path, split, feature, makeimg, clamp, makeJson, interpolation)


    datename = os.listdir(data_root)
    dataList = []
    gtList = []

    ground = '1'

    if gt_path == '':
        isGTExist = False
    else:
        isGTExist = True

    for dateCount in range(0, len(datename)):
        # lidar是光達的原始資料
        lidar = os.listdir(data_root + datename[dateCount] + '/level1')
        # if len(lidar) != 1: # 如果level1的資料夾裡有超過1個檔案就跳過
        #     dataList = []
        #     continue
        
        if isGTExist: ####### GT #######
                    gt_csv_path = gt_path + datename[dateCount] + '.csv'

                    with open(gt_csv_path, newline='') as gt_csvfile:
                        # 讀取 CSV 檔案內容
                        rows = csv.reader(gt_csvfile)
                        temp_gt = []
                        for i in rows:
                            temp_gt.append(i)

                    # 跳過NaN
                    count = 0
                    for asd in range(0, len(temp_gt)):
                        if temp_gt[count][1].lstrip() == 'NaN':
                            del temp_gt[count]
                            count = count - 1
                        count = count + 1

        
        # 開啟 CSV 檔案
        rows = []
        # for l in lidar:
        #     if l[-4:] != '.csv': # 如果不是.csv就跳過
        #         continue

        targetName = data_root + datename[dateCount] + '/level1/' + lidar[0]
        with open(targetName, newline='') as csvfile: # 讀資料
            next(csvfile)
            next(csvfile)

            # 讀取 CSV 檔案內容
            rows_ = csv.reader(csvfile)
            for i in rows_:
                rows.append(i)

        progress = tqdm(total=len(rows), desc=datename[dateCount])

        period = '0000' # 紀錄現在處理到哪一個時間區間，15分鐘一個間隔
        while True:
            gt = 0
            if isGTExist: ####### GT #######
                for index, data in enumerate(temp_gt):
                    if data[0].split()[1][0:2] + data[0].split()[1][3:5] == period:
                        gt = int(data[1].lstrip()[:-2])
                        break
            
            if not isGTExist or (isGTExist and gt != 0): # 不需要gt 或是gt不是NaN才繼續
                if period == '0000' and isCsvCompleteHead(rows):
                    targetindex = 0
                else:
                    targetindex = indexOfAPeriod(period, rows)
                    if targetindex == -1:
                        # print(f'skip period: {period}')
                        period = nextPeriod(period) # 下一個15分鐘
                        continue

                    if not isClosest(rows[targetindex][0].split()[1], rows[targetindex+1][0].split()[1]):
                        targetindex += 1

                if targetindex + 229 < len(rows):
                    for i in range(targetindex, targetindex+230):
                        progress.update(1)
                        # if rows[i][0][6:8] == '07' :    

                        if str(rows[i][1]) == '0' or str(rows[i][2]) == '0' or str(rows[i][3]) == '0': # temp=0, humi=0, press=0
                            isAbuse = True
                        if str(rows[i][4]) == 'V': # 跳過垂直的資料
                            continue

                        for j in range(0, 192): # 共192筆高度資料
                            if isGTExist: ####### GT #######
                                dict_ = {'time':rows[i][0], 'temp':rows[i][1], 'humi':rows[i][2], 'press':rows[i][3], 'dir':rows[i][4], 'height':j*30+60, 'rw':rows[i][7+j*4], 'snr':rows[i][8+j*4], 'sw':rows[i][9+j*4], 'si':rows[i][10+j*4], 'gt':int(gt)}
                            else:
                                dict_ = {'time':rows[i][0], 'temp':rows[i][1], 'humi':rows[i][2], 'press':rows[i][3], 'dir':rows[i][4], 'height':j*30+60, 'rw':rows[i][7+j*4], 'snr':rows[i][8+j*4], 'sw':rows[i][9+j*4], 'si':rows[i][10+j*4]}
                            dataList.append(dict_)
                        
                else:
                    print("targetindex + 229 !< len(rows)")
                    break
                
                # make img
                # global count_img
                # count_img += 1
                if makeimg:
                    MakeImg(dataList, period, ground, cfg) # dataList是15分鐘的資料 == 一張圖
                if isGTExist: gtList.append({'name':f"{dataList[0]['time'][:8]}{period}",'gt': gt})
                dataList = []

            # print(period)
            if int(period) >= 2345: # 最後一個區間處理結束
                # print(f"{datename[dateCount]} prosses end")
                break

            period = nextPeriod(period) # 下一個15分鐘

        dateCount += 1
        
    if makeJson: MakeJson(cfg, gtList=gtList)


if __name__ == '__main__':
    # testrun = True

    # test_data_root = '../test_data/raw/data/'
    # test_out_img_path = '../test_data/test_output/imgs/'
    # test_out_annotations_path = '../test_data/test_output/'
    # test_gt_path = '../test_data/raw/gt/'

    # data_root = test_data_root
    # out_img_path = test_out_img_path
    # out_annotations_path = test_out_annotations_path
    # gt_path = test_gt_path
    
    data_root = 'D:/LIDAR/RAW/'
    gt_path = 'D:/LIDAR/gt/'

    # data_root = '../test_data/missing_raw/raw/'
    # gt_path = '../test_data/missing_raw/gt/'
    # out_img_path = './output/missing/imgs/'
    # out_annotations_path = './output/missing/'


    out_img_path = 'D:/data/2-feature/184x192/'
    # out_annotations_path = '../test_data/annotation/'
    out_annotations_path = 'D:/data/annotations/'

    # cfg = Config(data_root=data_root, 
    #              out_img_path=out_img_path, 
    #              out_annotations_path=out_annotations_path, 
    #              gt_path=gt_path, 
    #              split=[], 
    #              feature=['rw', 'snr'], 
    #              makeimg=False, 
    #              clamp=True, 
    #              makeJson=True, 
    #              interpolation=False)

    t_start = time.time()
    ProcessRaw(data_root, out_img_path, out_annotations_path, gt_path=gt_path, split=[], feature=['rw','snr'], makeimg=True, clamp=False, makeJson=True, interpolation=False)
    t_end = time.time()

    print(f'Time spend: {t_end - t_start}')
    # print(f'count: {count_img}')


    




