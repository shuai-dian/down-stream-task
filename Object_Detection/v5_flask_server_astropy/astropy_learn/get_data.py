
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET
import os

leida = "leida"


def get_data(path1,i):
    filename = os.path.join(path1,i)
    new_filename=i.replace(".xml",".txt")

    in_file = open(filename)
    tree = ET.parse(in_file)
    root = tree.getroot()
    con_new_path = os.path.join(leida,new_filename)
    f = open(con_new_path,"w",encoding="utf-8")
    for data in root.iter("OSV"):
        # TAI = data.find("TAI").text
        UTC = data.find("UTC").text[4:]
        X = data.find("X").text
        Y = data.find("Y").text
        Z = data.find("Z").text
        VX = data.find("VX").text
        VY = data.find("VY").text
        VZ = data.find("VZ").text
        datas = UTC+","+X+","+Y+","+Z+","+VX+","+VY+","+VZ+"\n"
        print(datas)
        f.write(datas)
    f.close()

    #     # print(X)
    #     Xs.append(float(X))
    #     Ys.append(float(Y))
    #     Zs.append(float(Z))
    # return UTC,Xs,Ys,Zs,VX,VY

path1 = "SAR"
dirs = os.listdir(path1)
for i in dirs:
    data = get_data(path1,i)




