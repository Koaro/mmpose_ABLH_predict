import csv
import os
import numpy as np
from tqdm import tqdm

# csv_root = '../test_data/csv/'
csv_root = 'D:/csv/csv_without_vertical_33242/'
out_root = '../list_rw_snr/'

csv_names = os.listdir(csv_root)
csv_names.sort()

def FindRow(rows, gt):
    closest = 0
    height_list = []
    for i in range(len(rows)):
        height_list.append(int(rows[i][5]))

    height_list = np.array(height_list)
    height_list = height_list - int(gt)

    closest = (np.abs(height_list)).argmin()

    return closest

def Realheight(rows):
    for row in rows:
        n = (int(row[5]) - 60)/30
        real_h = 51 + 26 * n

        row[5] = real_h

out_list = []

progress = tqdm(total=len(csv_names), desc='read')
for csv_name in csv_names:
    file_name = os.path.join(csv_root, csv_name)
    with open(file_name, newline='') as file:
        next(file)
        # time, temp, humi, press, dir, height, rw, snr, sw, si, gt

        _rows = csv.reader(file)
        rows = list(_rows)

        Realheight(rows)

        gt = rows[0][10]
        time = rows[0][0]

        # print(f'{csv_name}, {gt}')

        closest_to_gt_i = FindRow(rows, gt)

        closest_to_gt = rows[closest_to_gt_i][5]
        closest_rw = rows[closest_to_gt_i][6]
        closest_snr = rows[closest_to_gt_i][7]

        up1rw = rows[closest_to_gt_i+1][6]
        up1snr = rows[closest_to_gt_i+1][7]
        up2rw = rows[closest_to_gt_i+2][6]
        up2snr = rows[closest_to_gt_i+2][7]
        
        down1rw = rows[closest_to_gt_i-1][6]
        down1snr = rows[closest_to_gt_i-1][7]
        down2rw = rows[closest_to_gt_i-2][6]
        down2snr = rows[closest_to_gt_i-2][7]

        out_list.append([time, gt, closest_to_gt, closest_rw, closest_snr, up1rw, up1snr, up2rw, up2snr, down1rw, down1snr,down2rw, down2snr])

        progress.update(1)



out_path = os.path.join(out_root, 'new2_list_aroundGT.csv')
with open(out_path, 'w', newline='') as file:
    writer = csv.writer(file)

    writer.writerow(['time', 'gt', 'closest to gt', 'closest rw', 'closest snr', '+1 rw', '+1 snr', '+2 rw', '+2 snr', '-1 rw', '-1 snr', '-2 rw', '-2 snr'])

    for row in out_list:
        writer.writerow(row)