# -*- coding:utf-8 -*-
import numpy
from matplotlib import pyplot as plt
import os
import struct
import base64

n_samples = 2480
# 1016
def parse_data(path):
    with open(path, 'rb') as file:
        while True:
            if len(file.read(n_samples * 2)) == 0:
                break
            else:
                datas = file.read(n_samples * 2)
                print(datas)
                for i in range(len(datas)):
                    print(datas[0:2])
                    if datas[0:2] == b'\x90\xeb' and datas[2:4] !=  b'\x90\xeb' :
                        self_msg = datas[2:8]
                        work_param = datas[8:42] # 雷达当前工作参数
                        target_message = datas[42:2470] # 数据处理输出的目标信息
                        software_ver = datas[2470:2478]# 雷达软件版本信息
                        W_detect = self_msg[0] # 信道分机 0：正常
                        print(self_msg)
                        print(work_param)
                        print(target_message)

                        # print(target_message[:2])
                        # hour = target_message[:2].hex()
                        # mi = target_message[2]
                        # sec = target_message[3]
                        # h_sec = target_message[:8]
                        # print(h_sec)
                        # data_time = hour +":"+str(mi)
                        #
                        # print("datetime",hour,mi,sec,h_sec)
                        # angel = target_message[8:10].decode('gb2312')
                        # print("angel:",angel)





                # print(type(datas[1]))
            break
        # while 1:
        #     datas = file.read()
        # imgs = np.fromfile(file, dtype=np.uint8).reshape(num, rows, cols)  # uint8
        # imgs = imgs.astype(np.float32) / 255.0
        # if flatten:
        #     imgs = imgs.reshape([num, -1])
    return None
filename = './redar.raw'
parse_data(filename)
# cdata = numpy.fromfile(filename, dtype='i8')  # '<':小端模式； 'i2':有符号16bits
# print(cdata)
# size = os.path.getsize(cdata)  # 获得文件大小
# print(size)
# # numpy.savetxt("cdata.txt", cdata, fmt='%d', delimiter=' ')  # 将全部数据写入txt文件便于查看
# print(size)%n_samples为每个chirp的采样点数，n_chirps为每帧chirp数
# xx = size / 2480  # 数据包帧数
# # # 读取数据包一帧数据
# # sdata = cdata[(xx - 1) * n_samples * n_chirps * rx * tx * 2:xx * n_samples * n_chirps * rx * tx * 2]
# sdata = sdata.reshape(32768, 8)  # 将一帧数据按数据包存储方式拆分八列
# # numpy.savetxt("sdata.txt", sdata, fmt='%d', delimiter=' ')  # 将一帧数据保存至txt文件便于查看
# # sdata1 = sdata.T  # 将sdata数组进行转置变为8行32768列
# # adcdata = sdata1[0:4] + 1j * sdata1[4:8]  # 实部虚部相加之后的adcdata数组为4行32768列复数形式
# # data_radar_1 = adcdata[0]
# #
# # # 一发一收下将一根天线一帧数据放置到data_radar数组下
# # data_radar_1 = data_radar_1.reshape(128, 256)
# # data_radar_1 = data_radar_1.T  # 此处进行了复共轭转置变为256行128列复数形式
# # data_radar = numpy.array([data_radar_1])  # 放置到三维数组中
# # data_radar = numpy.reshape(data_radar,(256,128,1))
# #
# # #  加汉宁窗进行距离fft
# # window = numpy.hanning(256)
# # for m in range(0,128):
# #     temp = data_radar[:, m, 0] * window
# #     temp_fft = numpy.fft.fft(temp, 256)
# #     data_radar[:,m,0] = temp_fft
# #     range_profile = numpy.reshape(data_radar,(256,128))
# #     range_profile_Temp =range_profile.T
# # #print(range_profile_Temp)
# #
# # a = 20*numpy.log10(abs(range_profile_Temp))
# # b = a.mean(axis=0)  #  按照每一列求平均值,一列共128个chirps
# #
# # #画图
# # plt .figure(1)
# # x = numpy.linspace(0,50,256)
# # y = b
# # plt.plot(x,y) #  目前显示的是一帧中256点对应的128chirp的均值
# # plt.show()
# # print('接收功率：',numpy.max(b))
# #
# #

# n_samples =6
# n_chirps = 2480
# n_Tx
# while True:/
    # data2 = fid.read(4960)
    # print(data2)
    # for i in range(len(data2)):
    #     print(data2[i])
    #     if data2[i] == b"eb" and data2[i+1] == b"90":
    #         head = data2[i+2:i+8]
    #         print(head)



# %n_samples为每个chirp的采样点数，n_chirps为每帧chirp数
# sdata = fid.read(n_samples*n_chirps*n_Rx*n_Tx*2,'int16');
# %通道解析
# fileSize = size(sdata, 1);
# lvds_data = zeros(1, fileSize/2);
# count = 1;
# for i=1:4:fileSize-5
#    lvds_data(1,count) = sdata(i) + 1i*sdata(i+2);
#    lvds_data(1,count+1) = sdata(i+1)+1i*sdata(i+3);  %IQ数据合并成复数
#    count = count + 2;
# end
# lvds_data = reshape(lvds_data, n_samples*n_RX, n_chirps);
# lvds_data = lvds_data.';
# cdata = zeros(n_RX,n_chirps*n_samples);
# for row = 1:n_RX       %天线个数
#   for i = 1: n_chirps     %一帧的chirp个数
#       cdata(row,(i-1)*n_samples+1:i*n_samples) = lvds_data(i,(row-1)*n_samples+1:row*n_samples);
#   end
# end
# fclose(fid);
# %4RX数据，n_chirps列，每i列为Chirp i的回波数据
# RX1data = reshape(cdata(1,:),n_samples,n_chirps);   %RX1数据
# RX2data = reshape(cdata(2,:),n_samples,n_chirps);   %RX2
# RX3data = reshape(cdata(3,:),n_samples,n_chirps);   %RX3
# RX4data = reshape(cdata(4,:),n_samples,n_chirps);   %RX4
