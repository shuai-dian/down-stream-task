# #方法一，利用关键字
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax1 = plt.axes(projection='3d')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import xml.etree.ElementTree as ET
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
def get_data(filename):
    in_file = open(filename)
    tree = ET.parse(in_file)
    root = tree.getroot()
    # Data_Block = root.attrib["OSV"]
    # print(Data_Block)
    Xs = []
    Ys = []
    Zs = []
    for data in root.iter("OSV"):
        TAI = data.find("TAI").text
        UTC = data.find("UTC").text
        X = data.find("X").text
        Y = data.find("Y").text
        Z = data.find("Z").text
        VX = data.find("VX").text
        VY = data.find("VY").text
        VZ = data.find("VZ").text
        # print(X)
        Xs.append(float(X))
        Ys.append(float(Y))
        Zs.append(float(Z))
    return Xs,Ys,Zs
fig=plt.figure()
ax2 = Axes3D(fig)
# z = np.linspace(0,13,1000)
# x = 5*np.sin(z)
# y = 5*np.cos(z)
# zd = 13*np.random.random(100)
# xd = 5*np.sin(zd)
# yd = 5*np.cos(zd)
# ax1.scatter3D(xd,yd,zd, cmap='Blues')  #绘制散点图
filepath = "data\S1B_OPER_AUX_POEORB_OPOD_20220524T084549_V20220503T225942_20220505T005942.EOF.xml"
x,y,z = get_data(filepath)
ax1.plot3D(x,y,z,'gray')    #绘制空间曲线
plt.show()




def get_data(filename):
    in_file = open(filename)
    tree = ET.parse(in_file)
    root = tree.getroot()
    # Data_Block = root.attrib["OSV"]
    # print(Data_Block)
    Xs = []
    Ys = []
    Zs = []
    for data in root.iter("OSV"):
        TAI = data.find("TAI").text
        UTC = data.find("UTC").text
        X = data.find("X").text
        Y = data.find("Y").text
        Z = data.find("Z").text
        VX = data.find("VX").text
        VY = data.find("VY").text
        VZ = data.find("VZ").text
        # print(X)
        Xs.append(X)
        Ys.append(Y)
        Zs.append(Z)
    return Xs,Ys,Zs

# fig = plt.figure()
# ax = plt.axes(projection ="3d")
# filepath = "data\S1B_OPER_AUX_POEORB_OPOD_20220524T084549_V20220503T225942_20220505T005942.EOF.xml"
# Xs,Ys,Zs = get_data(filepath)
#
#
# ax.scatter3D(Xs,Ys,Zs,cmap = "Greens")
# plt.show()




