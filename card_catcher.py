import pygetwindow as gw
import win32gui,win32ui,win32con
import cv2
import numpy as np
import time

import keyboard
from datetime import datetime
class WindowCapture:

    # properties
    w = 0
    h = 0
    hwnd = None
    cropped_x = 0
    cropped_y = 0
    offset_x = 0
    offset_y = 0

    # constructor
    def __init__(self):
        # find the handle for the window we want to capture
        titles = gw.getAllTitles()

        for t in titles:
            if "tx" in t.lower():
                self.hwnd = win32gui.FindWindow(None, t )
        if not self.hwnd:
            raise Exception('Window not found: ')

        # get the window size
        window_rect = win32gui.GetWindowRect(self.hwnd)
        self.w = window_rect[2] - window_rect[0]
        self.h = window_rect[3] - window_rect[1] 

        # account for the window border and titlebar and cut them off
        border_pixels = 8
        titlebar_pixels = 30
        self.w = self.w - (border_pixels * 2)
        self.h = self.h - titlebar_pixels - border_pixels
        self.cropped_x = border_pixels
        self.cropped_y = titlebar_pixels

        # set the cropped coordinates offset so we can translate screenshot
        # images into actual screen positions
        self.offset_x = window_rect[0] + self.cropped_x
        self.offset_y = window_rect[1] + self.cropped_y

    def get_screenshot(self):
        
        window_rect = win32gui.GetWindowRect(self.hwnd)
        self.w = window_rect[2] - window_rect[0]
        self.h = window_rect[3] - window_rect[1]

        # account for the window border and titlebar and cut them off
        border_pixels = 8
        titlebar_pixels = 30
        self.w = self.w - (border_pixels * 2)
        self.h = self.h - titlebar_pixels - border_pixels
        self.cropped_x = border_pixels
        self.cropped_y = titlebar_pixels

        # set the cropped coordinates offset so we can translate screenshot
        # images into actual screen positions
        self.offset_x = window_rect[0] + self.cropped_x
        self.offset_y = window_rect[1] + self.cropped_y
        # get the window image data
        wDC = win32gui.GetWindowDC(self.hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, self.w, self.h)
        cDC.SelectObject(dataBitMap)
        cDC.BitBlt((0, 0), (self.w, self.h), dcObj, (self.cropped_x, self.cropped_y), win32con.SRCCOPY)

        # convert the raw data into a format opencv can read
        #dataBitMap.SaveBitmapFile(cDC, 'debug.bmp')
        signedIntsArray = dataBitMap.GetBitmapBits(True)
        img = np.fromstring(signedIntsArray, dtype='uint8')
        img.shape = (self.h, self.w, 4)

        # free resources
        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())

        # drop the alpha channel, or cv.matchTemplate() will throw an error like:
        #   error: (-215:Assertion failed) (depth == CV_8U || depth == CV_32F) && type == _templ.type() 
        #   && _img.dims() <= 2 in function 'cv::matchTemplate'
        img = img[...,:3]

        # make image C_CONTIGUOUS to avoid errors that look like:
        #   File ... in draw_rectangles
        #   TypeError: an integer is required (got type tuple)
        # see the discussion here:
        # https://github.com/opencv/opencv/issues/14866#issuecomment-580207109
        img = np.ascontiguousarray(img)

        return img

    # find the name of the window you're interested in.
    # once you have it, update window_capture()
    # https://stackoverflow.com/questions/55547940/how-to-get-a-list-of-the-name-of-every-open-window
    def list_window_names(self):
        def winEnumHandler(hwnd, ctx):
            if win32gui.IsWindowVisible(hwnd):
                print(hex(hwnd), win32gui.GetWindowText(hwnd))
        win32gui.EnumWindows(winEnumHandler, None)

    # translate a pixel position on a screenshot image to a pixel position on the screen.
    # pos = (x, y)
    # WARNING: if you move the window being captured after execution is started, this will
    # return incorrect coordinates, because the window position is only calculated in
    # the __init__ constructor.
    def get_screen_position(self, pos):
        return (pos[0] + self.offset_x, pos[1] + self.offset_y)


def image_resize(image, width = None, height = None, inter = cv2.INTER_CUBIC):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim,fx=4,fy=4 ,interpolation = inter)
    # return the resized image
    return resized

def create_template(suit, number):
    im_prefix = 'images/cards_m'
    # 数字の左右の位置調整。pixel
    shift = {
        's': [-1, -1, 0, -1, -1, 0, -1, -1, -1, 0, -1, 0, 0],
        'h': [-1, -1, 0, -1, -1, 0, -1, -1, -1, 0, -1, 0, 0],
        'd': [-1, -1, 0, -1, -1, 0, -1, -1, -1, 0, -1, 0, 0],
        'c': [-1, -1, 0, -1, -1, 0, -1, -1, -1, 0, -1, 0, 0],
    }
    # space = {'s': 4, 'h': 6, 'd': 4, 'c': 2}
    space = {'s': 8, 'h': 10, 'd': 6, 'c': 8}
    color = {'s': 'b', 'h': 'r', 'd': 'r', 'c': 'b'}
    n = cv2.imread('{}/{}{}.png'.format(im_prefix, number, color[suit]))
    s = cv2.imread('{}/{}.png'.format(im_prefix, suit))[:-8, :]
    
    # Initiate SIFT detector
   
    ###
    s_width = s.shape[1]
    s_height = s.shape[0]
    # 番号部は位置調整用の余白を2px追加
    template_width = max(s.shape[1], n.shape[1] + 2)

    n_back = np.full((n.shape[0], template_width, 3), 255)
    s_back = np.full((s_height, template_width, 3), 255)
    spacer = np.full((space.get(suit), template_width, 3), 255)

    n_width = n.shape[1]
    n_left_margin = int((n_back.shape[1] - n_width)/2) + shift[suit][number - 1]
    s_left_margin = int((s_back.shape[1] - s_width)/2)

    n_back[:, n_left_margin:n_left_margin + n.shape[1]] = n
    s_back[:, s_left_margin:s_left_margin + s.shape[1]] = s

    n_and_s = np.vstack((n_back, spacer, s_back)).astype('u1')
    template = cv2.cvtColor(n_and_s, cv2.COLOR_BGR2GRAY)
    return template

    # downscale
    # scaled = cv2.resize(template, (int(n_and_s.shape[1]*3/5), int(n_and_s.shape[0]*3/5)), interpolation=cv2.INTER_AREA)
    # return scaled

def temp_size(suit,number,image):
    print("hallå")
    im_prefix = 'images/cards_m'
    # 数字の左右の位置調整。pixel
    shift = {
        's': [-1, -1, 0, -1, -1, 0, -1, -1, -1, 0, -1, 0, 0],
        'h': [-1, -1, 0, -1, -1, 0, -1, -1, -1, 0, -1, 0, 0],
        'd': [-1, -1, 0, -1, -1, 0, -1, -1, -1, 0, -1, 0, 0],
        'c': [-1, -1, 0, -1, -1, 0, -1, -1, -1, 0, -1, 0, 0],
    }
    # space = {'s': 4, 'h': 6, 'd': 4, 'c': 2}
    space = {'s': 8, 'h': 10, 'd': 6, 'c': 8}
    color = {'s': 'b', 'h': 'r', 'd': 'r', 'c': 'b'}
    n = cv2.imread('{}/{}{}.png'.format(im_prefix, number, color[suit]))
    s = cv2.imread('{}/{}.png'.format(im_prefix, suit))[:-8, :]
    

    sift = cv2.SIFT()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(n,None)
    kp2, des2 = sift.detectAndCompute(s,None)
    kp3,des3 = sift.detectAndCompute(image,None)
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    num_matches = flann.knnMatch(des3,des1,k=2)
    suit_matches = flann.knnMatch(des3,des1,k=2)
    print("fitta")
    print(num_matches,suit_matches,"va")
    num_good = []
    for m,n in num_matches:
        if m.distance < 0.7*n.distance:
            num_good.append(m)
            num_good = []
    suit_good = []
    for m,n in suit_matches:
        if m.distance < 0.7*n.distance:
            suit_good.append(m)


    return num_good,suit_good

def test():
    img1  = cv2.imread("D:\\GITHUB REPOS\\Playing-Card-Recognition\\gamla test o train\\test.png")
    img2 = cv2.imread("D:\\GITHUB REPOS\\Playing-Card-Recognition\\gamla test o train\\Kh.png")
  
    sift = cv2.SIFT()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    print("bök")

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    else:
        print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = None
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = matchesMask, # draw only inliers
                    flags = 2)

    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    print("nä")
    plt.imshow(img3, 'gray'),plt.show()
    return img3
        
def main():
    # out_image = cv2.imread('images_for_test/poker.png')
    """
    active_window_info = get_active_window_info()

    if active_window_info['kCGWindowOwnerName'] != 'PokerStars':
        print('Active window is not PokerStars. exited.')
        exit(0)

    out_image = capture_window(active_window_info)
    """
    #out_image = cv2.imread("D:\\GITHUB REPOS\\Playing-Card-Recognition\\gamla test o train\\test.png")
    out_image = cv2.imread("images_for_test\poker.png")
    cv2.imshow("hej",out_image)
    cv2.waitKey(0)    
    h,w,s= out_image.shape
    print(h,w,"shape")
    #out_image = image_resize(out_image,width=int(w/2))
    if h != 1358:
        h = 1358
        #yolo = image_resize(out_image,width=w)

    elif w != 1906:
        w = 1906
        yolo = image_resize(out_image,height=h)


    
    #cv2.imwrite("halfpkr.png",yolo)
    # 画面の真ん中あたりを取得
    hh, ww = out_image.shape[:-1]
    hh = int(hh)
    ww = int(ww)
    #ww = int(ww/3)
   # hh = int(hh/3)

   # out_image = out_image[hh:hh+int(hh * 1.1), ww:ww+ww]
    window_image = cv2.cvtColor(out_image, cv2.COLOR_BGR2GRAY)

    # cv2.imwrite(f'out_{int(datetime.now().timestamp())}.png', out_image)

    # # scale
    # target_width = 2192
    # scale = target_width/out_image.shape[1]
    # width = int(out_image.shape[1] * scale)
    # height = int(out_image.shape[0] * scale)
    # dim = (width, height)
    #
    # out_image = cv2.resize(out_image, dim, interpolation=cv2.INTER_AREA)
    # window_image = cv2.resize(window_image, dim, interpolation=cv2.INTER_AREA)

    detected_cards = []
    
    for s in 'shdc':
        for i in range(1,14):
            template = create_template(s, i)
            result = cv2.matchTemplate(window_image, template, cv2.TM_CCOEFF_NORMED)
            threshold = 0.8
            w, h = template.shape[:: -1]
            # Store the coordinates of matched area in a numpy array 
            loc = np.where( result >= threshold)  
            # cv2.imwrite('template_{}_{}.png'.format(i, s), template)
            for pt in zip(*loc[:: -1]):
                cv2.rectangle(out_image, pt, (pt[0] + w, pt[1] + h),(0, 255, 255), 2 )
                #here i wanted to move the mouse to the coordinates of a found item, however
                #i cant get these two right ↓        ↓
            # 検出結果から検出領域の位置を取得
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            # print('s={}, i={}'.format(s, i))
            # print('min_val={:.3f}, max_val={:.3f}, min_loc={}, max_loc={}'.format(min_val, max_val, min_loc, max_loc))

            if max_val < 0.95:
                continue

            detected_cards.append([s, i, max_val])

            # top_left = max_loc
            # w, h = template.shape[::-1]
            # bottom_right = (top_left[0] + w, top_left[1] + h)
            #
            # cv2.rectangle(out_image, top_left, bottom_right, (255, 0, 0), 2)
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    img = mpimg.imread('halfpkr.png')
    imgplot = plt.imshow(img)
    
    print("snopp")
    import time

    time.sleep(10)
    print("lul")
main()