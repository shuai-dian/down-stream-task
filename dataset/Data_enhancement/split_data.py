from sklearn.model_selection import train_test_split
import os
import shutil
train_data = "train"
img_data = os.path.join(train_data,"images")
lab_data = os.path.join(train_data,"labels")
list_data = os.listdir(img_data)
train_set, test_set = train_test_split(list_data, test_size=0.05, random_state=42)



print(len(test_set))
for i in test_set:
    old_img_t = os.path.join("train",'images',i)
    old_lab_t = os.path.join("train","labels", i[:-4] + ".txt")

    img_t = os.path.join("val",'images',i)
    lab_t = os.path.join("val","labels",i[:-4] + ".txt")

    shutil.move(old_img_t,img_t)
    shutil.move(old_lab_t,lab_t)

