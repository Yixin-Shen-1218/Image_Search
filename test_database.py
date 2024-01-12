import json
import os

database = []
dirs = os.listdir("flowers/")
print(dirs)

# save the images path combined with labels
for dir_name in dirs:
    path = f"flowers/{dir_name}"
    files = os.listdir(path)
    files.sort(key=lambda x: int(x.split('_')[0]))
    for file_ in files[:10]:
        if not os.path.isdir(path + file_):
            f_name = str(file_)
            img_label = {f_name: dir_name}
            database.append(img_label)

print(database)
with open('static/img_label/label.json', 'w', encoding='utf-8') as f:
    json.dump(database, f, ensure_ascii=False, indent=4)
