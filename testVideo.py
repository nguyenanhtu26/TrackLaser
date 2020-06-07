
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


# Doc file video (khoảng 60 khung/s)
cap = cv2.VideoCapture("video/test_6t7.mp4")

# Khai báo tạo video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
outVideo = cv2.VideoWriter('output.avi', fourcc, 5, (1280, 720))

blackLower = (0, 0, 60)
blackUpper = (255, 255, 120)

whiteLower = (0, 0, 210)
whiteUpper = (255, 255, 255)

cXPre = 0
cYPre = 0
frameThu = 0
vel = 0
new_width = 600
new_height = 400

while True:
    # Read a new frame
    ok, frame = cap.read()
    if not ok:
        # Neu khong doc duoc tiep thi out
        break
    else:
        frameThu += 1

    fileNameOrig = 'anhOrig%s.jpg' % (frameThu)
    print(fileNameOrig)
    cv2.imwrite(fileNameOrig, frame)

    ratio = frame.shape[0] / 500.0  # Chiều cao ảnh chuẩn hóa(chia cho 500)
    orig = frame.copy()
    frame = imutils.resize(frame, height=500)

    # convert the image to grayscale, blur it, and find edges
    # in the image
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 75, 200)

    mask = cv2.inRange(hsv, blackLower, blackUpper)

    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    # show the original image and the edge detected image
    # print("STEP 1: Edge Detection")
    # cv2.imshow("Image", frame)
    # cv2.imshow("Edged", edged)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Step 2: Finding Contours
    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    cnts = cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:]
    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break

    if len(approx) != 4:
        print("Không thấy tờ giấy trong frame thứ ", frameThu)

    # show the contour (outline) of the piece of paper
    # print("STEP 2: Find contours of paper")
    # cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    # cv2.imshow("Outline", frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Step 3: Apply a Perspective Transform & Threshold
    # apply the four point transform to obtain a top-down
    # view of the original image
    try:
        warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
        # convert the warped image to grayscale, then threshold it
        # to give it that 'black and white' paper effect
        # warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        # T = threshold_local(warped, 11, offset=10, method="gaussian")
        # warped = (warped > T).astype("uint8") * 255
        # show the original and scanned images
        fileNameImage = 'anhOut%s.jpg' % (frameThu)
        print(fileNameImage)
        cv2.imwrite(fileNameImage, warped)
    except:
        print("Lỗi ở frame thứ", frameThu)

    # STEP 4: Detect point laser coordinates
    warped = cv2.resize(src=warped, dsize=(new_width, new_height))

    blurredWar = cv2.GaussianBlur(warped, (5, 5), 0)
    hsvWar = cv2.cvtColor(blurredWar, cv2.COLOR_BGR2HSV)

    maskWar = cv2.inRange(hsvWar, whiteLower, whiteUpper)

    maskWar = cv2.erode(maskWar, None, iterations=2)
    maskWar = cv2.dilate(maskWar, None, iterations=2)  # Đang là ảnh Gray với 2 mức xám 0 và 255

    # convert the grayscale image to binary image
    ret, thresh = cv2.threshold(maskWar, 127, 255, 0)

    # calculate moments of binary image
    M = cv2.moments(thresh)

    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    # put text and highlight the center
    print("Tọa độ tâm: ", cX, cY)
    # print("STEP 4: Detect point laser coordinates")

    # STEP 5: Velocity calculation of laser point
    if (cYPre != 0 and cYPre != 0):
        dis = np.sqrt(((cX - cXPre) ** 2) + ((cY - cYPre) ** 2))
        vel = dis * 60  # Because 60fps
    cYPre = cY
    cXPre = cX

    # STEP 6: Show results
    print("Vận tốc: ", vel)

    # key = cv2.waitKey(1) & 0xff  # Neu nhan q thi thoat
    # if key == ord('q'):
    #     break

# Hiện thị video
cv2.destroyAllWindows()
