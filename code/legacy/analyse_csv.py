label_csv_dir = "/home/jaklimczak/envs/FinalProject/resources/dataset/train_labels.csv"
import csv

def count_ones_and_zeros(csv_file, column_name):
    ones_count = 0
    zeros_count = 0

    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            value = row[column_name]
            if value == '1':
                ones_count += 1
            elif value == '0':
                zeros_count += 1

    return ones_count, zeros_count

column_to_count = 'MGMT_value'

ones, zeros = count_ones_and_zeros(label_csv_dir, column_to_count)

print(f"Number of 1s: {ones}")
print(f"Number of 0s: {zeros}")
