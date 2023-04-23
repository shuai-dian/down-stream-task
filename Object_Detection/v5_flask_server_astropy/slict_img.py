import cv2
import os
import numpy as np
def slice_im(image0,sliceHeight=1024, sliceWidth=1024):
    win_h, win_w = image0.shape[:2]
    print(win_h, win_w)
    print(image0.shape)
    splits = []
    imgs = []
    for y0 in range(0, image0.shape[0], sliceHeight):
        for x0 in range(0, image0.shape[1], sliceWidth):
            # n_ims += 1
            #这一步确保了不会出现比要切的图像小的图，其实是通过调整最后的overlop来实现的
            #举例:h=6000,w=8192.若使用640来切图,overlop:0.2*640=128,间隔就为512.所以小图的左上角坐标的纵坐标y0依次为:
            #:0,512,1024,....,5120,接下来并非为5632,因为5632+640>6000,所以y0=6000-640
            if y0 + sliceHeight > image0.shape[0]:
                y = image0.shape[0] - sliceHeight
            else:
                y = y0

            if x0 + sliceWidth > image0.shape[1]:
                x = image0.shape[1] - sliceWidth
            else:
                x = x0
            splits.append([x,y])
            img = image0[x:x + 1024, y:y + 1024, :]
            imgs.append(img)
            # bbox = []
            # if bbox != []:
            #     new_boxes = []
            #     for i in bbox:
            #         real_bbox = [x + i[0],y + i[1],x + i[2] ,y + i[3],"sat",i[5].tolist()]
            #         print(real_bbox)
            #         new_boxes.append(real_bbox)

    from PIL import Image
    background = Image.new('RGB', size=(win_h, win_w))
    for i in range(4):
        img = Image.fromarray(cv2.cvtColor(imgs[i],cv2.COLOR_BGR2RGB))
        if i == 1:
            background.paste(img, box=splits[2])
        elif i == 2 :
            background.paste(img, box=splits[1])
        else:
            background.paste(img, box=splits[i])
    background.show()

            # cv2.imshow("img{}_{}".format(str(x),str(y)), img)
    # cv2.waitKey(0)
            # slice_xmax = x + sliceWidth
            # slice_ymax = y + sliceHeight
            # exiset_obj_list=exist_objs([x,y,slice_xmax,slice_ymax],object_list, sliceHeight, sliceWidth)
            # exiset_obj_list = exist_objs_iou([x,y,slice_xmax,slice_ymax],object_list, sliceHeight, sliceWidth, win_h, win_w)
            # if exiset_obj_list !=[]:  # 如果为空,说明切出来的这一张图不存在目标
                # extract image
                # window_c = image0[y:y + sliceHeight, x:x + sliceWidth]
                # get black and white image
            #     window = cv2.cvtColor(window_c, cv2.COLOR_BGR2GRAY)
            #
            #     # find threshold that's not black
            #     #
            #     ret, thresh1 = cv2.threshold(window, 2, 255, cv2.THRESH_BINARY)
            #     non_zero_counts = cv2.countNonZero(thresh1)
            #     zero_counts = win_size - non_zero_counts
            #     zero_frac = float(zero_counts) / win_size
            #     # print "zero_frac", zero_fra
            #     # skip if image is mostly empty
            #     if zero_frac >= zero_frac_thresh:
            #         if verbose:
            #             print("Zero frac too high at:", zero_frac)
            #         continue
            #         # else save
            #     else:
            #         outpath = os.path.join(outdir, out_name + \
            #                                '|' + str(y) + '_' + str(x) + '_' + str(sliceHeight) + '_' + str(sliceWidth) + \
            #                                '_' + str(pad) + '_' + str(win_w) + '_' + str(win_h) + ext)
            #
            #         #
            #         cnt += 1
            #         # if verbose:
            #         #     print("outpath:", outpath)
            #         cv2.imwrite(outpath, window_c)
            #         n_ims_nonull += 1
            #         #------制作新的xml------
            #         make_slice_voc(outpath,exiset_obj_list,sliceHeight,sliceWidth)


image0 = cv2.imread("sat_00000.0101_out.jpg")
img = slice_im(image0)