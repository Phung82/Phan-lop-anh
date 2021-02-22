import numpy as np
import cv2
import pickle

def nothing(x):
    pass

def undistort(img, cal_dir='./bin/cal_pickle.p'):
    with open(cal_dir, mode='rb') as f:
        file = pickle.load(f)
    mtx = file['mtx']
    dist = file['dist']
    #dùng undistort để cải thiện sự biến dạng của anh do ống kính của camera
    #Chuyển đổi hình ảnh để bù cho hiện tượng méo ống kính
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst

def colorFilter(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lowerYellow = np.array([18,94,140])
    upperYellow = np.array([48,255,255])
    lowerWhite = np.array([0, 0, 200])
    upperWhite = np.array([255, 255, 255])
    #inRange dùng để phát hiện một đối tượng dựa trên phạm vi giá trị pixel trong không gian màu HSV
    # Tạo 2 lớp Mask trắng và vàng
    maskedWhite= cv2.inRange(hsv,lowerWhite,upperWhite)
    #cv2.imshow("mask-trang ",maskedWhite)
    maskedYellow = cv2.inRange(hsv, lowerYellow, upperYellow)
    #cv2.imshow("mask-vang",maskedYellow)
    #Tính toán độ lệch bit theo từng phần tử của hai mảng màu
    combinedImage = cv2.bitwise_or(maskedWhite,maskedYellow)
    #cv2.imshow("cv2.bitwise_or", combinedImage)
    return combinedImage


def thresholding(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5,5))
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 0)
    imgCanny = cv2.Canny(imgBlur, 50, 100)
    #imgClose = cv2.morphologyEx(imgCanny, cv2.MORPH_CLOSE, np.ones((10,10)))
    #Làm giãn nở một hình ảnh với số lần giản là 1
    imgDial = cv2.dilate(imgCanny,kernel,iterations=1)
    #áp dụng kỹ thuật xói mòn ảnh với độ mòn là 1
    imgErode = cv2.erode(imgDial,kernel,iterations=1)
    #Lọc ảnh màu
    imgColor = colorFilter(img)
    #cv2.imshow("imgColor", imgColor)
    combinedImage = cv2.bitwise_or(imgColor, imgErode)
    #cv2.imshow("cv2.bitwise_or", combinedImage)
    return combinedImage,imgCanny,imgColor

def intialCotrol(intialCotrol):
    cv2.namedWindow("Control")
    cv2.resizeWindow("Control", 400, 200)
    #Tạo trackbars để đặt phạm vi giá trị HSV
    cv2.createTrackbar("Width Top", "Control", intialCotrol[0],50, nothing)
    cv2.createTrackbar("Height Top", "Control", intialCotrol[1], 100, nothing)
    cv2.createTrackbar("Width Bottom", "Control", intialCotrol[2], 50, nothing)
    cv2.createTrackbar("Height Bottom", "Control", intialCotrol[3], 100, nothing)


    #hàm xác định tỷ lệ  các điểm thể hiện góc nhìn của camera
def valTrackbars():
    widthTop = cv2.getTrackbarPos("Width Top", "Control")
    heightTop = cv2.getTrackbarPos("Height Top", "Control")
    widthBottom = cv2.getTrackbarPos("Width Bottom", "Control")
    heightBottom = cv2.getTrackbarPos("Height Bottom", "Control")

    #xác định các điểm thể hiện góc nhìn của camera
    #các thông sô tỷ lệ  có thể thay đổi được nhờ vào bảng điều khiển
    src = np.float32([(widthTop/100,heightTop/100), (1-(widthTop/100), heightTop/100),
                    (widthBottom/100, heightBottom/100), (1-(widthBottom/100), heightBottom/100)])

    #print("SRC", src)
    #các thông số tỷ được cố định
    #src = np.float32([(0.42 0.63), (0.58,0.63), (0.14,0.87), (0.86,0.87)])
    return src

    #vẽ các điểm thể hiện góc nhìn của camera lên ảnh kết quả trả về là 1 khung hình
def drawPoints(img,src):
    #lấy chiều dài và chiệu rộng của khung hình
    img_size = np.float32([(img.shape[1],img.shape[0])])
    #print("Shape 0",img.shape[0])
    #src = np.float32([(0.42 0.63), (0.58,0.63), (0.14,0.87), (0.86,0.87)])
    #lấy lại vị trí thật trên khung hình
    #cv2.FILLED vẽ đầy
    src = src * img_size
    #print("src: ", src)
    for x in range( 0,4):
        cv2.circle(img,(int(src[x][0]),int(src[x][1])),10,(0,255,0),cv2.FILLED)
    return img

def pipeline(img, s_thresh=(100, 255), sx_thresh=(15, 255)):
    img = undistort(img)
    img = np.copy(img)
    # Chuyển đổi sang không gian màu HLS và tách kênh V
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    h_channel = hls[:, :, 0]
    #Lấy đạo hàm tại x để làm nổi bậc các đường thẳng nằm ngang dùng phân vùng 
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 1)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    #Xác định ngưỡng độ dốc
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Xác định ngưỡng các kênh màu
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary

#hàm chuyển đổi góc nhìn sang góc nhìn từ trên xuống
def perspective_warp(img,
                    #kích thước cảu ảnh đầu ra
                     dst_size=(1280, 720),
                     src=np.float32([(0.42,0.63),(0.58,0.63),(0.14,0.87),(0.86,0.87)]),
                     dst=np.float32([(0,0), (1, 0), (0,1), (1,1)])):
    img_size = np.float32([(img.shape[1],img.shape[0])])
    src = src* img_size

    dst = dst * np.float32(dst_size)
    # Cho điểm src và dst, tính ma trận biến đổi phối cảnh
    M = cv2.getPerspectiveTransform(src, dst)
    #print("M",M)
    #Áp dụng một chuyển đổi phối cảnh cho hình ảnh
    warped = cv2.warpPerspective(img, M, dst_size)
    #print("Warped", warped)
    return warped

def get_hist(img):
    hist = np.sum(img[img.shape[0]//2:,:], axis=0)
    #print("hist", hist)
    return hist


left_a, left_b, left_c = [], [], []
right_a, right_b, right_c = [], [], []

#hàm xác định làn đường

def sliding_window(img, nwindows=15, margin=50, minpix=1, draw_windows=True):

    global left_a, left_b, left_c, right_a, right_b, right_c
    left_fit_ = np.empty(3)
    right_fit_ = np.empty(3)
    out_img = np.dstack((img, img, img)) * 255

    #lấy bma trận histogram của ảnh
    histogram = get_hist(img)
    #print("Histogram", histogram)
    
    # tìm đỉnh nửa trái và phải
    midpoint = int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Thiết lập chiều cao cửa sổ
    window_height = np.int(img.shape[0] / nwindows)
    # Xác định vị trí x và y của tất cả các pixel khác không trong ảnh
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Vị trí hiện tại sẽ được cập nhật cho mỗi cửa sổ
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Tạo danh sách trống để nhận chỉ số pixel làn đường bên trái và bên phải
    left_lane_inds = []
    right_lane_inds = []

    # duyệt qua các ô trên các khung hình của histogram
    for window in range(nwindows):
        # Xác định ranh giới cửa sổ theo x và y (và phải và trái)
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Vẽ các ô trên hình ảnh trực quan
        if draw_windows == True:
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                          (100, 255, 255), 1)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                          (100, 255, 255), 1)
            # Xác định các pixel khác không trong x và y trong cửa sổ
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Nối các chỉ số này vào danh sách
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # Nếu tìm thấy> minpix pixel thì đặt lại cửa sổ tiếp theo vào vị trí trung bình của chúng
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Nối các mảng của các chỉ số
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Trích xuất các vị trí pixel dòng trái và phải
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    if leftx.size and rightx.size:
        # Ghép một đa thức bậc hai cho mỗi bên làn đường
        # đa thức dạng ax^2+bX+c
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        left_a.append(left_fit[0])
        left_b.append(left_fit[1])
        left_c.append(left_fit[2])

        right_a.append(right_fit[0])
        right_b.append(right_fit[1])
        right_c.append(right_fit[2])

        #lấy 10 giá trị cuối cùng để nhận là kết quả 
        left_fit_[0] = np.mean(left_a[-10:])
        left_fit_[1] = np.mean(left_b[-10:])
        left_fit_[2] = np.mean(left_c[-10:])

        right_fit_[0] = np.mean(right_a[-10:])
        right_fit_[1] = np.mean(right_b[-10:])
        right_fit_[2] = np.mean(right_c[-10:])


        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])

        left_fitx = left_fit_[0] * ploty ** 2 + left_fit_[1] * ploty + left_fit_[2]
        right_fitx = right_fit_[0] * ploty ** 2 + right_fit_[1] * ploty + right_fit_[2]

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 100]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 100, 255]

        return out_img, (left_fitx, right_fitx), (left_fit_, right_fit_), ploty
    else:
        return img,(0,0),(0,0),0


#dùng công thức bán kính đường cong bằng phương pháp đạo hàm
#https://vi.wikipedia.org/wiki/B%C3%A1n_k%C3%ADnh_cong

def get_curve(img, leftx, rightx):
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    y_eval = np.max(ploty)
    # xác định đơn vị mét trên mỗi pixel theo chiều y
    ym_per_pix = 1 / img.shape[0]
    # xác định đơn vị mét trên mỗi pixel theo chiều x
    xm_per_pix = 0.1 / img.shape[0]

    # Điều chỉnh đa thức mới cho x, y trong không gian 
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
    # Tính bán kính mới của độ cong theo công thức tin bán kính bằng đạo hàm
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    car_pos = img.shape[1] / 2
    l_fit_x_int = left_fit_cr[0] * img.shape[0] ** 2 + left_fit_cr[1] * img.shape[0] + left_fit_cr[2]
    r_fit_x_int = right_fit_cr[0] * img.shape[0] ** 2 + right_fit_cr[1] * img.shape[0] + right_fit_cr[2]
    lane_center_position = (r_fit_x_int + l_fit_x_int) / 2
    center = (car_pos - lane_center_position) * xm_per_pix / 10
    # Trả lại bán kính của đường cong là met

    return (l_fit_x_int, r_fit_x_int, center)

#chuyển đổi góc nhìn
def inv_perspective_warp(img,
                     dst_size=(1280,720),
                     src=np.float32([(0,0), (1, 0), (0,1), (1,1)]),
                     dst=np.float32([(0.43,0.65),(0.58,0.65),(0.1,1),(1,1)])):
    img_size = np.float32([(img.shape[1],img.shape[0])])
    src = src* img_size
    # Xác định lại các điểm đến
    dst = dst * np.float32(dst_size)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, dst_size)
    return warped

#hàm vẽ làn đường mô phỏng khi đã được nhận dạng
def draw_lanes(img, left_fit, right_fit,frameWidth,frameHeight,src):
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    color_img = np.zeros_like(img)

    left = np.array([np.transpose(np.vstack([left_fit, ploty]))])
    right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))])
    points = np.hstack((left, right))

    cv2.fillPoly(color_img, np.int_(points), (0, 255,0))
    inv_perspective = inv_perspective_warp(color_img,(frameWidth,frameHeight),dst=src)
    inv_perspective = cv2.addWeighted(img, 0.5, inv_perspective, 0.7, 0)
    return inv_perspective

def drawLines(img,lane_curve):
    myWidth = img.shape[1]
    myHeight = img.shape[0]
    #print(myWidth,myHeight)
    cv2.line(img, (int(lane_curve // 100) + myWidth // 2, myHeight - 30),
             (int(lane_curve // 100) + myWidth // 2, myHeight), (0,0,255), 3)


    return img
#hàm chuyển đổi matraan số và qui định vị trí hiển thị các khung hình
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver



