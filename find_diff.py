import os
from PIL import Image

path = 'output/test/imgs/'
path2 = 'D:/csv/csv_without_vertical'

files = os.listdir(path)
files.sort()

cut_files = []
for f in files:
    cut_files.append(f[:-4])

files2 = os.listdir(path2)
files2.sort()

cut_files2 = []
for f in files2:
    cut_files2.append(f[:-4])

print(cut_files[:6])
print(cut_files2[:6])

# dif = list(set(cut_files2) - set(cut_files))
dif = list(set(cut_files) - set(cut_files2))
dif.sort()
print(len(dif))

os.makedirs("output/test/dif/", exist_ok=True)
for d in dif:
    img = Image.open(path+d+'.png')
    img.save(f"output/test/dif/{d}.png")

# with open('output/dif.txt', 'w') as file:
#     for d in dif:
#         file.write(d+'\n')