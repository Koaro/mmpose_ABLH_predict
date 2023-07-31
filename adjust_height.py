import csv

csv_path = '../list_rw_snr/list_aroundGT.csv'

rows = {}

with open(csv_path, newline='') as file:
    rows = csv.DictReader(file)
    rows  = list(rows)

for row in rows:
    n = (int(row['closest to gt']) - 60)/30
    real_h = 51 + 26 * n

    row['closest to gt'] = real_h

out_csv_path = '../list_rw_snr/new_list_aroundGT.csv'
with open(out_csv_path, 'w', newline='') as file:
    fieldnames = ['time', 'gt', 'closest to gt', 'closest rw', 'closest snr', '+1 rw', '+1 snr', '+2 rw', '+2 snr', '-1 rw', '-1 snr', '-2 rw', '-2 snr']

    writer = csv.DictWriter(file, fieldnames=fieldnames)

    writer.writeheader()

    for row in rows:
        writer.writerow(row)