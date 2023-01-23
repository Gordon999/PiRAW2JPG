#!/usr/bin/env python3
import cv2
import numpy as np
import glob
import os

# initial values
gamma_value = 1.2
contrast    = 1.0
saturation  = 1.6
brightness  = 1.0

def gammaCorrection(src, gamma):
    invGamma = 1 / gamma
    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)
    return cv2.LUT(src, table)

# setup directories
Home_Files  = []
Home_Files.append(os.getlogin())

# find raw files
files = glob.glob("/home/" + Home_Files[0]+ "/Pictures/*.raw")
files.sort()
valid = 0

if len(files) > 0:
    for x in range(0,len(files)):
        # Open raw file
        f = open(files[x],'rb')
        image = np.fromfile (f,dtype=np.uint8,count=-1)
        f.close()
        # check size
        if image.size == 1658880:  #Pi3 1536x864
            cols = 1536
            rows = 864
            valid = 1
        elif image.size == 14929920: #Pi3 4608x2592
            cols = 4608
            rows = 2592
            valid = 1
        elif image.size == 3732480: #Pi3 2304x1296
            cols = 2304
            rows = 1296
            valid = 1
        elif image.size == 384000:  #Pi2 640x480
            cols = 640
            rows = 480
            valid = 1
        elif image.size == 2562560: #Pi2 800x600
            cols = 1664
            rows = 1232
            valid = 1
        elif image.size == 10171392: #Pi2 3280x2464
            cols = 3280
            rows = 2464
            valid = 1
        elif image.size == 2592000:  #Pi2 1920x1080
            cols = 1920
            rows = 1080
            valid = 1
        elif image.size == 1586304:  #Pi1 1296x972
            cols = 1296
            rows = 972
            valid = 1
        elif image.size == 4669440:  #PiHQ 2028x1520
            cols = 2048
            rows = 1520
            valid = 2
        elif image.size == 3317760:  #PiHQ 2028x1080
            cols = 2048
            rows = 1080
            valid = 2
        elif image.size == 18580480:  #PiHQ 4056x3040
            cols = 4056
            rows = 3040
            valid = 2
        else:
            valid = 0
            print("Failed to find suitable file ",files[x])
        # process if correct size
        if valid > 0:
            if image.size == 10171392:
                image = image.reshape(int(image.size/4128),4128)
                for j in range(4127,4099,-1):
                    image  = np.delete(image, j, 1)
            if image.size == 18580480:
                image = image.reshape(int(image.size/6112),6112)
                for j in range(6111,6083,-1):
                    image  = np.delete(image, j, 1)
            if image.size == 1586304:
                image = image.reshape(int(image.size/1632),1632)
                for j in range(1631,1619,-1):
                    image  = np.delete(image, j, 1)
            # extract data
            if valid == 1:
                A = image.reshape(int(image.size/5),5)
                A  = np.delete(A, 4, 1)
            else:
                A = image.reshape(int(image.size/3),3)
                A  = np.delete(A, 2, 1)
            F  = A.reshape(rows,cols)
            C  = A.reshape(int(rows/2),int(cols*2))
            D  = np.split(C, 2, axis=1)
            H  = D[0].reshape(int(D[0].size/2),2)
            I  = np.split(H, 2, axis=1)
            b  = I[0].reshape(int(rows/2),int(cols/2))
            g0 = I[1].reshape(int(rows/2),int(cols/2))
            L  = D[1].reshape(int(D[0].size/2),2)
            M  = np.split(L, 2, axis=1)
            g1 = M[0].reshape(int(rows/2),int(cols/2))
            r  = M[1].reshape(int(rows/2),int(cols/2))

            # some colour correction
            Red   = r * 1
            Blue  = b * 1
            Green = ((g0/2) + (g1/2)) * 0.7

            # combine B,G,R
            BGR=np.dstack((Blue,Green,Red)).astype(np.uint8)
            res = cv2.resize(BGR, dsize=(cols,rows), interpolation=cv2.INTER_CUBIC)

            # split res into Y,Cr,Cb
            res_rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
            res_ycbcr = cv2.cvtColor(res_rgb, cv2.COLOR_RGB2YCrCb)
            Y, Cr, Cb = cv2.split(res_ycbcr)

            # split rgb into Y,Cr,Cb
            rgb = cv2.cvtColor(F, cv2.COLOR_BGR2RGB)
            ycbcr = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
            Y1, Cr1, Cb1 = cv2.split(ycbcr)

            # combine Y1 from rgb and Cr an Cb from res
            image_merge = cv2.merge([Y1,Cr,Cb])
            img_bgr = cv2.cvtColor(image_merge, cv2.COLOR_YCrCb2BGR)

            hsvImg = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2HSV)

            #multiple by a factor to change the saturation
            hsvImg[...,1] = hsvImg[...,1]*saturation

            #multiple by a factor of less than 1 to reduce the brightness 
            hsvImg[...,2] = hsvImg[...,2]*brightness

            img_bgr=cv2.cvtColor(hsvImg,cv2.COLOR_HSV2BGR)

            # adjust gamma
            img_bgr = gammaCorrection(img_bgr, gamma_value)

            # adjust contrast
            lab= cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=contrast, tileGridSize=(8,8))
            cl = clahe.apply(l_channel)
            limg = cv2.merge((cl,a,b))
            img_bgr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
              
            # save output
            fname = files[x].split('.')
            cv2.imwrite(fname[0] + ".jpg", img_bgr)

            # show result
            result = cv2.resize(img_bgr, dsize=(int(cols/2),int(rows/2)), interpolation=cv2.INTER_CUBIC)
            cv2.imshow('output',result)
    
        # wait for a key press
        #cv2.waitKey()
    cv2.destroyAllWindows()

