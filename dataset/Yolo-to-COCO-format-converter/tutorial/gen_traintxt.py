import os
root_p = "train"
img_list = os.listdir(root_p)

f = open("test.txt","w")
for i in img_list:
    # print(i)
    if i.split(".")[-1] != "txt":
        img_cp ="tutorial\\" + root_p + "\\" + i
        f.write(img_cp + "\n")

f.close()


