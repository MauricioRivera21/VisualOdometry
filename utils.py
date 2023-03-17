import numpy as np 
import matplotlib.pyplot as plt 
import cv2 as cv
import os 


"""
VIDEO 
"""

def variacion_error(gt_poses, vo_poses, i):
    if i>1:
        gt  = np.array(gt_poses)
        vo  = np.array(vo_poses)
        error = np.linalg.norm(gt - vo, axis=1)
        var_error = error[-1] - error[-2]
    else:
        var_error = 0.1
    return var_error


def plot_error(gt_poses, vo_poses,path):
    dicc_axis = { 'KITTI_sequence_1':[-15,15,-10,60], 
                  'KITTI_sequence_2':[-5,50,-10,30] }

    gt  = np.array(gt_poses)
    vo  = np.array(vo_poses)
    error = np.linalg.norm(gt - vo, axis=1)
    
    xe,ye = vo.T
    xg,yg = gt.T

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figheight(4)
    fig.set_figwidth(8)
    fig.suptitle('Resultados de Monocular Estimation')

    ax1.plot(xe, ye, 'b')
    ax1.plot(xg, yg, 'r')
    
    ax1.axis(dicc_axis[path])
    ax1.set(xlabel='x(m)', ylabel='y(m)')
    ax1.grid(True)

    ax2.plot(error)
    ax2.set(xlabel='Iterations', ylabel='Error(m)')
    ax2.grid(True)

    plt.show()
    


def play_video(dir_path,odometry,SHOW_FEATURES = 0,PROCESS = 0):

    win_name = "Trip"

    for i in range(50):
    
        frame = odometry.images[i]

        if PROCESS:
            frame, _ = proyect_feature_processing(frame, var_error=0, umbral=200)

        text = 'frame #' + str(1+i) + "/50" 
        font = cv.FONT_HERSHEY_SIMPLEX
        org = (20,50)
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2

        if SHOW_FEATURES:
            kp1,kp2,des1,des2, _ = odometry.extract_features(i, var_error=0, umbral=200)
            q1,q2             = odometry.process_Frame(kp1,kp2,des1,des2)

            output_image = cv.drawKeypoints(frame, kp2, 0, (0, 255, 0))
        else:
            output_image = frame

        image = cv.putText(output_image, text, org, font, fontScale, 
                    color, thickness, cv.LINE_AA, False)

        cv.imshow(win_name, image)

        key = cv.waitKey(100)
        if key == 27:  # ESC
            break

    cv.destroyWindow(win_name)

"""
FEATURES
"""
# def mascara(img_shape):
#     mask = np.ones(img_shape)
#     temp = int(img_shape[0]/2)
#     mask[temp:] = 0
#     mask = np.uint8(mask)
#     return mask

def threshold_filter(img, umbral):
    img_p1, th1 = cv.threshold(img,umbral,255,cv.THRESH_BINARY)
    return th1

def threshold_adaptive_filter(img, umbral):
    hist,bins = np.histogram(img.flatten(),256,[0,256])
    bright_bits = sum(hist[umbral:])
    total_bits = sum(hist[0:])
    comparacion_cantidad_bits = (bright_bits/total_bits)*100
    #print("inicial", comparacion_cantidad_bits)

    #se busca que la cantidad de bits que se detecten sea 9.4% del total o menos
    while comparacion_cantidad_bits >= 9.4 and umbral <= 200:
        umbral = umbral + 2
        hist,bins = np.histogram(img.flatten(),256,[0,256])
        bright_bits = sum(hist[umbral:])
        comparacion_cantidad_bits = (bright_bits/total_bits)*100

    #se busca que la cantidad de bits que se detecten sea 9.4% del total o mas
    while comparacion_cantidad_bits <= 9.4:
        umbral = umbral - 2        #disminuir el umbral, aumenta comparacion_cantidad_bits 
        hist,bins = np.histogram(img.flatten(),256,[0,256])
        bright_bits = sum(hist[umbral:])
        comparacion_cantidad_bits = (bright_bits/total_bits)*100
    
    img_p1, th1 = cv.threshold(img,umbral,255,cv.THRESH_BINARY)
    return th1, umbral

def canny(img):
    img = cv.Canny(img, 100, 200)
    #img = cv.Canny(img, 230, 250)
    return img

# def prewitt_kernel_X(img):
#     Prewitt_kernel_X = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
#     Prewitt_img_X = cv.filter2D(img,-1,Prewitt_kernel_X)
#     return Prewitt_img_X

def dilatacion(img):
    kernel_cruz = np.ones((3,3),np.uint8)
    kernel_cruz[0][0]=0
    kernel_cruz[0][2]=0
    kernel_cruz[2][0]=0
    kernel_cruz[2][2]=0
    img = cv.dilate(img,kernel_cruz, iterations=1)
    return img

def erosion(img):
    kernel_cruz = np.ones((3,3),np.uint8)
    kernel_cruz[0][0]=0
    kernel_cruz[0][2]=0
    kernel_cruz[2][0]=0
    kernel_cruz[2][2]=0
    img = cv.erode(img,kernel_cruz, iterations=1)
    return img

# def Hough_trans(img):
#   # Detect points that form a line
#   max_slider = 100
#   lines = cv.HoughLinesP(img, 1, np.pi/180, max_slider, minLineLength= 10, maxLineGap= 250)
#   # Draw lines on the image
#   for line in lines:
#       x1, y1, x2, y2 = line[0]
#       cv.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
#   return lines

def blurring_filter(img):
  img = cv.bilateralFilter(img, 3, 5, 5)
  return img


def proyect_feature_processing(img, var_error, umbral):
    # umbral = 150
    # img = cv.resize(img, (640,480))
    # img_shape = (img.shape[0],img.shape[1])  
    # mask = mascara(img_shape)

    # img = prewitt_kernel_X(img)
    # img = dilatacion(img)    
    # img = threshold_filter(img, umbral)
    # img = blurring_filter(img)
    # img = canny(img)
    # img = img*mask
    #lines = Hough_trans(img)

    if var_error >= 0.05:
        img, umbral = threshold_adaptive_filter(img, umbral)
    else:
        img = threshold_filter(img, umbral)
    
    img = blurring_filter(img)
    img = canny(img)


    return img, umbral
