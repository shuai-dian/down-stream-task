import os
img_cp = "train/images"
lab_cp = "train/labels"


cont = {}

for i in os.listdir(img_cp):
    img_c = os.path.join(img_cp,i)
    lab_c = os.path.join(lab_cp,i[:-4] + ".txt")
    #new_lab_c = os.path.join("val/new_labels",i)
    f = open(lab_c,"r")
    #ff = open(new_lab_c,"w")
    data = f.readlines()
    f.close()

    if len(data) == 0:
        # print(True)
        print(lab_c)
        print(img_c)
        os.remove(lab_c)
        os.remove(img_c)
        # img_cp = os.path.join("val","images",i[:-4] + )

    # for d in data:
    #     cls = d.split()[0]
    #     if cls in cont:
    #         cont[cls] += 1
    #     else:
    #         cont[cls] = 0


        #if cls == "18":
        #    pass
        #else:
        #    ff.write(d)
            
    #ff.close()
print(cont)
