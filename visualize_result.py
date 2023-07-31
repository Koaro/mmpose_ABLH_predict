from PIL import Image, ImageDraw
import os
import pandas as pd
from tqdm import tqdm

img_root = 'D:/data/2-feature/128x128/'
out_root = 'C:/Users/lab602/OneDrive/桌面/研究所/實驗資料/Atomos/compare_new/error/'
out_dir = 'hourglass_128x128_6d_2f_2stack_test_record_3k'
xlsx_path = f'C:/Users/lab602/OneDrive/桌面/研究所/實驗資料/Atomos/compare_new/學長的訓練資料/{out_dir}.xlsx'
out_path = os.path.join(out_root, f"vis_{out_dir}")

if not os.path.exists(out_path):
    os.makedirs(out_path)

df = pd.read_excel(xlsx_path, engine="openpyxl")
predict_dic = df.to_dict('records')

# print(predict_dic[0]['date'].strftime('%Y%m%d%H%M'))
for predict in tqdm(predict_dic):
    img_name = predict['date'].strftime('%Y%m%d%H%M')
    img_path = os.path.join(img_root, f"{img_name}.png")
    img = Image.open(img_path)

    draw = ImageDraw.Draw(img)
    
    x = img.size[0]/2
    y_gt = predict['gt']
    y_predict = predict['level']

    draw.ellipse(xy=(x-2.5, y_gt-2.5, x+2.5, y_gt+2.5), fill=(255,0,0))
    draw.ellipse(xy=(x-2.5, y_predict-2.5, x+2.5, y_predict+2.5), fill=(0,0,255))

    img.save(os.path.join(out_path, f"vis_{img_name}.png"))






