import cv2
import numpy as np
### Ham Tim Toa Do cua cac duong luoi
def GridCoordinates(image):
    # Input: Anh sau khi da tach khung
    # Output: Vecto toa do cua duong ke doc
    #          Vecto toa do cua duong ke ngang
    img = image.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Threshold the image
    ret, thres = cv2.threshold(img, 127, 255, 0)

    # Step 1: Create an empty skeleton
    size = np.size(thres)
    skel = np.zeros(thres.shape, np.uint8)

    # Get a Cross Shaped Kernel
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    # Repeat steps 2-4
    while True:
        # Step 2: Open the image
        open = cv2.morphologyEx(thres, cv2.MORPH_OPEN, element)
        # Step 3: Substract open from the original image
        temp = cv2.subtract(thres, open)
        # Step 4: Erode the original image and refine the skeleton
        eroded = cv2.erode(thres, element)
        skel = cv2.bitwise_or(skel, temp)
        thres = eroded.copy()
        # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
        if cv2.countNonZero(thres) == 0:
            break
    # Displaying the final skeleton
    cv2.imshow("Skeleton Image", skel)
    # [x, y] = FindCenterLaserPointer(image)
    [x, y] = FindCenter(image)
    numVerDir = y - 50
    verDir = skel[numVerDir, :]
    left = 0
    right = 0
    verCoor = []
    i = 0
    while i < len(verDir):
        if verDir[i] != 0:
            left = i
            i += 1
            while verDir[i] != 0:
                i += 1
                continue
            right = i - 1
            center = int((left + right) / 2)
            cv2.circle(img, (center, numVerDir), 1, (0, 0, 255), 1, cv2.LINE_AA)
            verCoor.append(center)
        else:
            i += 1
    print(verCoor)
    numHonDir = int((verCoor[0] + verCoor[1]) / 2)
    honDir = skel[:, numHonDir]
    top = 0
    bottom = 0
    honCoor = []
    i = 0
    while i < len(honDir):
        if honDir[i] != 0:
            top = i
            i += 1
            while honDir[i] != 0:
                i += 1
                continue
            bottom = i - 1
            center = int((top + bottom) / 2)
            cv2.circle(img, (numHonDir, center), 1, (0, 0, 255), 1, cv2.LINE_AA)
            honCoor.append(center)
        else:
            i += 1
    print(honCoor)
    # Displaying the image with central points
    # cv2.imshow("Vertical Point", img)
    return verCoor, honCoor
### Ham tim Top - Right - Bottom - Left
def FindTRBL(values):
    # Input: Ma tran can tim T - R - B - L
    # Output: Gia tri cua Tmost - Rmost - Bmost - Lmost

    leftmost = 0
    rightmost = 0
    topmost = 0
    bottommost = 0
    temp = 0
    for i in range(np.size(values, 1)):
        col = values[:, i]
        if np.sum(col) != 0.0:
            rightmost = i
            if temp == 0:
                leftmost = i
                temp = 1
    for j in range(np.size(values, 0)):
        row = values[j, :]
        if np.sum(row) != 0.0:
            bottommost = j
            if temp == 1:
                topmost = j
                temp = 2
    return [topmost, bottommost, leftmost, rightmost]

### Ham xac dinh toa do trung tam cua laser Pointer
def FindCenterLaserPointer(image):
    # Input: Anh sau khi da tach khung
    # Output: Toa do trung tam cua diem Laser Pointer
    frame = image.copy()
    # STEP 1: Filter with red => Eliminate the area < 10
    LASER_MIN = np.array([0, 0, 250], np.uint8)
    LASER_MAX = np.array([255, 255, 255], np.uint8)

    while (True):
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_threshed = cv2.inRange(hsv_img, LASER_MIN, LASER_MAX)
        cv2.imshow("Frame Thresh 1", frame_threshed)

        # Eliminate the regions which have the area smaller 10
        _, contours, _ = cv2.findContours(frame_threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if (np.size(contours) <= 2):
            break
        for cnt in contours:
            area = cv2.contourArea(cnt)
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            x = approx.ravel()[0]
            y = approx.ravel()[1]
            if area < 10:
                cv2.circle(frame, (x, y), 2, (0, 0, 0), 2, cv2.LINE_AA) # To mau den nhung vung co mau do co S < 10
                # cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 0, 0), 10, cv2.LINE_AA)
    cv2.imshow("Boi den", frame)
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_threshed = cv2.inRange(hsv_img, LASER_MIN, LASER_MAX)
    cv2.imshow("Frame Thresh 2", frame_threshed)


    # STEP 2: Determine the edge of laser center
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    canny = cv2.Canny(blurred, 50, 150, 1)
    cv2.imshow("Canny", canny)

    # Focusing on [top right bottom left] of red region
    [top, bottom, left, right] = FindTRBL(frame_threshed)
    # print(top, bottom, left, right)
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)
    cv2.imshow("Image after filter RED", frame)

    sectionFrame = canny[top:bottom, left:right]  # Select the region which has red
    [topmost, bottommost, leftmost, rightmost] = FindTRBL(sectionFrame)
    # print(topmost, bottommost, leftmost, rightmost)
    laserx = left + round((rightmost + leftmost) / 2)
    lasery = top + round((bottommost + topmost) / 2)
    # print(laserx, lasery)
    return laserx, lasery
### Ham tinh toan ra vi tri thuc cua Laser Poiter
def RealCoordinatesOfLaserPointer(image, x, y, verCoor, honCoor):
    # Input: x, y: Toa do cua diem laser
    #        verCoor: Toa do cua cac truc doc
    #        honCoor: Toa do cua ca truc ngang
    #        scale: Ty le quy doi (cm/pixels)
    # Output: [x_real, y_real]: Toa do thuc te cua diem laser
    img = image.copy()
    size_y = np.size(img, 0)
    size_x = np.size(img, 1)
    scale_x = 27.4/size_x
    scale_y = 20/size_y
    font = cv2.FONT_HERSHEY_COMPLEX
    x_real = 0
    y_real = 0
    # Duyet theo hang ngang
    delta = verCoor - np.ones(np.size(verCoor))*x
    minValue = min(abs(delta))
    for i in range(len(delta) - 1):
        if np.sign(delta[i]) != np.sign(delta[i + 1]):
            # Tinh khoang cach den i
            x_real = round((x - verCoor[i])*scale_x + i, 2)
            cv2.line(img, (x, y), (verCoor[0], y), (0, 0, 255), 1)
            cv2.putText(img, str(x_real) + 'cm', (verCoor[0], y - 30), font, 1, (0, 255, 255))
        if abs(delta[i]) == minValue:
            if minValue == 0:
                x_real = i
            break
    ### Duyet theo hang doc
    y_real = round((honCoor[1] - y)*scale_y, 2)
    cv2.line(img, (x, y), (x, honCoor[1]), (0, 0, 255), 1)
    cv2.putText(img, str(y_real) + 'cm', (x + 30, honCoor[1]), font, 1, (0, 255, 255))
    cv2.imshow("Detected Image", img)
    return x_real, y_real, img

### Ham xac dinh toa do trung tam cua laser Pointer
def FindCenter(image):
    # Input: Anh dau vao da tach khung
    # Output: Toa do (x, y) cua laser pointer
    img = image.copy()
    whiteLower = (0, 0, 230)
    whiteUpper = (255, 255, 255)
    blurredWar = cv2.GaussianBlur(img, (5, 5), 0)
    hsvWar = cv2.cvtColor(blurredWar, cv2.COLOR_BGR2HSV)

    maskWar = cv2.inRange(hsvWar, whiteLower, whiteUpper)
    cv2.imshow("hsvWar1", maskWar)
    maskWar = cv2.dilate(maskWar, None, iterations=2)  # Đang là ảnh Gray với 2 mức xám 0 và 255
    maskWar = cv2.erode(maskWar, None, iterations=2)

    cv2.imshow("hsvWar2", maskWar)
    # convert the grayscale image to binary image
    ret, thresh = cv2.threshold(maskWar, 127, 255, 0)
    cv2.imshow("hsvWar3", thresh)
    # calculate moments of binary image
    M = cv2.moments(thresh)

    # calculate x,y coordinate of center
    cX = 0
    cY = 0
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    return cX, cY