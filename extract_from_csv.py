import csv
import os
import pandas as pd

file_path = '../compare/128X128/128_6d_2feature_no_SHM_400elr'
gt_list_path = '../gt_list_test_LIDAR.csv'
out_path = f'{file_path}_test_extract.csv'

source_file = {'date':[], 'pblh':[], 'level':[]}
gt_list = {'date':[], 'pblh':[], 'gt':[]}
output =[]


### read from xlsx ###
xlsx_file = pd.read_excel(f"{file_path}.xlsx", engine='openpyxl').to_dict()

for i in range(len(xlsx_file['date'])):
        date = xlsx_file['date'][i]
        y = date.strftime('%Y')
        m = int(date.strftime('%m'))
        d = int(date.strftime('%d'))
        h = date.strftime('%H')
        min = date.strftime('%M')


        source_file['date'].append(f"{y}/{m}/{d} {h}:{min}")
        source_file['pblh'].append(xlsx_file['pblh'][i])
        source_file['level'].append(xlsx_file['level'][i])
#     print(f"date: [{source_file['date'][i]}], pblh: [{source_file['pblh'][i]}], level: [{source_file['level'][i]}]")

# for i in range(6):
#      print(f"{i}: {source_file['date'][i]}")

# quit()


'''
### read from csv ###
with open(f"{file_path}.csv", newline='') as file:
    rows = csv.DictReader(file)

    for i, row in enumerate(rows):
        source_file['date'].append(row['date'])
        source_file['pblh'].append(row['pblh'])
        source_file['level'].append(row['level'])
# for i in range(6):
#     print(f"date: [{source_file['date'][i]}], pblh: [{source_file['pblh'][i]}], level: [{source_file['level'][i]}]")'''


### gt ###
with open(gt_list_path, newline='') as file:
    rows = csv.DictReader(file)

    for i, row in enumerate(rows):
        gt_list['date'].append(row['date'])
        gt_list['pblh'].append(row['pblh'])
        gt_list['gt'].append(row['gt'])
# for i in range(6):
#     print(f"date: [{gt_list['date'][i]}], pblh: [{gt_list['pblh'][i]}], gt: [{gt_list['gt'][i]}]")

for i in range(len(gt_list['date'])):
    try:
        index = source_file['date'].index(gt_list['date'][i])
        output.append([source_file['date'][index], source_file['pblh'][index], source_file['level'][index]])
    
    except ValueError:
        print(f"date: [{gt_list['date'][i]}] not found.")

print(f"len of output: [{len(output)}]")
        
with open(out_path, 'w', newline='') as outfile:
    writer = csv.writer(outfile)

    writer.writerow(['date', 'pblh', 'level'])

    for row in output:
        writer.writerow(row)


