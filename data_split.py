import os

csv_path = r""

with open(csv_path, "r", encoding="utf-8") as f:
    csv_data = f.readlines()
    columns = csv_data.pop(0)

    split_ratio = int(len(csv_data) * 0.1)

    valid_data = csv_data[:split_ratio]
    valid_data.insert(0, columns)

    train_data = csv_data[split_ratio:]
    train_data.insert(0, columns)

save_path = r""
train_save_path = os.path.join(save_path, "train.csv")
with open(train_save_path, "w", encoding="utf-8") as f:
    train_data = "".join(train_data)
    f.write(train_data)

valid_save_path = os.path.join(save_path, "valid.csv")
with open(valid_save_path, "w", encoding="utf-8") as f:
    valid_data = "".join(valid_data)
    f.write(valid_data)
