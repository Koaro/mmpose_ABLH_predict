# 把lidar資料轉成每15分鐘的資料，並做成圖片

import csv
import os
import numpy as np
from tqdm import tqdm
import json

from PIL import Image
import math

import time

test_data_root = '../test_data/raw/data/'
test_gt_path = '../test_data/raw/gt/'

test_out_img_path = './output/test/imgs/'
test_out_annotations_path = './output/test/'

isPreData = False
isAbuse = False

testrun = False

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


count_img = 0 # debug 用
def ProcessRaw(data_root, out_img_path, out_annotations_path, gt_path = '', split=[], feature=[], makeimg=True, clamp=False ):

    datename = os.listdir(data_root)
    dataList = []
    gtList = []

    ground = '1'

    if gt_path == '':
        isGTExist = False
    else:
        isGTExist = True


    rwlist = []
    snrlist = []
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
        while True: # 處理一筆RAW檔
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

                if int(period[2:4]) == 0:
                    for i in range(0, 192, 2):
                        rwlist.append(dataList[i]["rw"])
                        snrlist.append(dataList[i]["snr"])


                if isGTExist: gtList.append({'name':f"{dataList[0]['time'][:8]}{period}",'gt': gt})
                dataList = []

            # print(period)
            if int(period) >= 2345: # 最後一個區間處理結束
                # print(f"{datename[dateCount]} prosses end")
                break

            period = nextPeriod(period) # 下一個15分鐘

        dateCount += 1
    
    rwlist.sort()
    snrlist.sort()

    with open(f"list_rw_snr/list_rw_snr.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["rw", "snr"])

        for i in range(len(rwlist)):
            writer.writerow([rwlist[i], snrlist[i]])

if __name__ == '__main__':
    # testrun = True
    data_root = test_data_root
    gt_path = test_gt_path
    out_img_path = test_out_img_path
    out_annotations_path = test_out_annotations_path
    
    data_root = 'D:/LIDAR/RAW/'
    gt_path = 'D:/LIDAR/gt/'
    # out_img_path = './output/test/imgs/'
    # out_annotations_path = './output/test/'

    # data_root = '../test_data/missing_raw/raw/'
    # gt_path = '../test_data/missing_raw/gt/'
    # out_img_path = './output/missing/imgs/'
    # out_annotations_path = './output/missing/'


    out_img_path = 'D:/data/2-feature/33242/'
    # out_annotations_path = '../test_data/annotation/'
    out_annotations_path = 'D:/data/annotations/'

    t_start = time.time()
    ProcessRaw(data_root, out_img_path, out_annotations_path, gt_path=gt_path, split=[], feature=['rw', 'snr'], makeimg=True, clamp=True)
    t_end = time.time()

    print(f'Time spend: {t_end - t_start}')
    # print(f'count: {count_img}')


    




