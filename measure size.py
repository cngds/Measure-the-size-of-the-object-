import cv2
from scipy.spatial.distance import euclidean

#1cm代表多少个像素
pixel_per_cm = 0

#记录鼠标左键按下的坐标
click_point_x = 0
click_point_y = 0

#计算标志，=1说明坐标更新
calculate_flag = 0

#回调函数，将x，y坐标赋值到click_point等对应的变量。用于后面判断鼠标点击的坐标在不在轮廓里。
def event_lbutton(event, x, y, flags, param):
    global click_point_x, click_point_y, calculate_flag
    if event == cv2.EVENT_LBUTTONDOWN:
        click_point_x = x
        click_point_y = y
        calculate_flag = 1
       

#计算pixel_per_cm，默认校准图像是1cmx1cm的阴影方块。
def calculate_pixel_per_cm(tl, tr):
    dist_in_pixel = euclidean(tl, tr)
    dist_in_cm = 1
    pixel_p_cm = dist_in_pixel/dist_in_cm
    return pixel_p_cm


#处理图像的函数，转灰度图，还有高斯模糊（核需要奇数表示）。找出边缘，然后筛选出小于一定数量的边缘，return回去
def img_to_contours(img, gauss = 5, minarea = 100):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if (gauss % 2) == 0:
        print("错误，gauss参数不是奇数！")
        return []
    
    blur = cv2.GaussianBlur(gray, (gauss,gauss), 0)
    retval, dst = cv2.threshold(blur, 0, 255,  cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)

    # Find the contours of the objects in the image
    contours, _ = cv2.findContours(dst.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    #过滤小于一定值的边缘。剩下的都是大于一定值的边缘
    return_contours = []
    for x in contours:
        if cv2.contourArea(x) > minarea:
            return_contours.append(x)
    
    return return_contours
    
if __name__ == '__main__':
    


    cap = cv2.VideoCapture(2)
    #用的是200万像素的摄像头。这里设置1920*1080大小。
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", event_lbutton)

    while True:
        # Load the image
        _, img = cap.read()
        contours = img_to_contours(img)
        
        # Loop through the contours and calculate the area of each object
        for cnt in contours:
            box = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(box)
            
            #box对应的四个point,可能是随机的。就是说box[0]有时候是top left的，有时候是top right的角。
            #固定(tl, tr, br, bl) = box表达还是有点草率？
            (tl, tr, br, bl) = box
            if calculate_flag == 1:
                if cv2.pointPolygonTest(cnt, (click_point_x, click_point_y), 0) == 1:
                    pixel_per_cm = calculate_pixel_per_cm(tl, tr)
                    calculate_flag = 0
                    
            cv2.drawContours(img, [box.astype("int")], -1, (0, 0, 255), 2)
            mid_pt_horizontal = (tl[0] + int(abs(tr[0] - tl[0])/2), tl[1] + int(abs(tr[1] - tl[1])/2))
            mid_pt_verticle = (tr[0] + int(abs(tr[0] - br[0])/2), tr[1] + int(abs(tr[1] - br[1])/2))
            
            if pixel_per_cm != 0:
                wid = euclidean(tl, tr)/pixel_per_cm
                ht = euclidean(tr, br)/pixel_per_cm
                cv2.putText(img, "{:.3f}cm".format(wid), (int(mid_pt_horizontal[0]), int(mid_pt_horizontal[1])), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                cv2.putText(img, "{:.3f}cm".format(ht), (int(mid_pt_verticle[0]), int(mid_pt_verticle[1])), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Show the final image
        cv2.imshow('image', img)

        # quit the program if you press 'q' on keyboard
        if cv2.waitKey(1) == ord("q"):
            break
        
    # closing the camera
    cap.release()
    cv2.destroyAllWindows()

    
