# import the necessary packages

from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils


# Four_point_transform
def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")  # Ma trận 0 kích thước 4x2, kiểu float32
    # Ma trận tương ứng là tọa độ của điểm  trên-trái, trên-phải, dưới-trái, dưới-phải
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)  # Tính tổng theo hàng
    rect[0] = pts[np.argmin(s)]  # Trên-trái
    rect[2] = pts[np.argmax(s)]  # Dưới-phải
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)  # Tìm 4 góc từ hình nhận được
    (tl, tr, br, bl) = rect  # Lấy 4 góc
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
    dst = np.array([  # Tạo mảng 4x2 cách điểm cần căn chỉnh
        [0, 0],  # tl
        [maxWidth - 1, 0],  # tr
        [maxWidth - 1, maxHeight - 1],  # br
        [0, maxHeight - 1]], dtype="float32")  # bl
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required = True,
# 	help = "Path to the image to be scanned")
# args = vars(ap.parse_args())

blackLower = (0, 0, 60)
blackUpper = (255, 255, 120)

new_width = 600
new_height = 400

whiteLower = (0, 0, 210)
whiteUpper = (255, 255, 255)

# Step 1: Edge Detection
# load the image and compute the ratio of the old height
# to the new height, clone it, and resize it
image = cv2.imread("anhOrig125.jpg")
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height=500)
# convert the image to grayscale, blur it, and find edges
# in the image

blurred = cv2.GaussianBlur(image, (5, 5), 0)
hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(gray, 75, 200)

mask = cv2.inRange(hsv, blackLower, blackUpper)

mask = cv2.erode(mask, None, iterations=2)
mask = cv2.dilate(mask, None, iterations=2)

# gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
# edged = cv2.Canny(gray, 75, 200)

# res = cv2.bitwise_and(hsv, hsv,  mask= mask)

# show the original image and the edge detected image
print("STEP 1: Edge Detection")
# cv2.imshow("Image", image)
# cv2.imshow("Edged", mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Step 2: Finding Contours
# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
cnts = cv2.findContours(mask.copy(), cv2.RETR_LIST,
                        cv2.CHAIN_APPROX_SIMPLE)  # Trả về bao gồm điểm trong và ngoài của đường viền và list candy
cnts = imutils.grab_contours(cnts)  # Lấy điểm đường viền
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[
       :]  # sắp xếp để loại bỏ các chấm nhỏ và 4 giá trị đầu (vì là hình chữ nhật)

# loop over the contours
for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c,
                         True)  # Hàm tính toán độ dài đường cong hoặc chu vi đường viền kín với điều kiện đường cong đóng
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)  # Tính xấp xỉ đường cong đa giác với độ chính xác biết trước
    # if our approximated contour has four points, then we
    # can assume that we have found our screen
    if len(approx) == 4:
        screenCnt = approx
        break

# show the contour (outline) of the piece of paper
print("STEP 2: Find contours of paper")
# cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
# cv2.imshow("Outline", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Step 3: Apply a Perspective Transform & Threshold
# apply the four point transform to obtain a top-down
# view of the original image
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
# convert the warped image to grayscale, then threshold it
# to give it that 'black and white' paper effect
# warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
# show the original and scanned images
print("STEP 3: Apply perspective transform")
# cv2.imshow("Original", imutils.resize(orig, height=650))
# cv2.imshow("Scanned", imutils.resize(warped, height=650))
# cv2.waitKey(0)

# STEP 4: Detect point laser coordinates
warped = cv2.resize(src=warped, dsize=(new_width, new_height))

blurredWar = cv2.GaussianBlur(warped, (5, 5), 0)
hsvWar = cv2.cvtColor(blurredWar, cv2.COLOR_BGR2HSV)

maskWar = cv2.inRange(hsvWar, whiteLower, whiteUpper)

maskWar = cv2.erode(maskWar, None, iterations=2)
maskWar = cv2.dilate(maskWar, None, iterations=2)  # Đang là ảnh Gray với 2 mức xám 0 và 255

# # convert the grayscale image to binary image
# ret,thresh = cv2.threshold(maskWar,127,255,0)
#
# # calculate moments of binary image
# M = cv2.moments(thresh)
#
# # calculate x,y coordinate of center
# cX = int(M["m10"] / M["m00"])
# cY = int(M["m01"] / M["m00"])
#
# # put text and highlight the center
# print(cX, cY)

print("STEP 4: Detect point laser coordinates")
cv2.imshow("Image wraped", warped)
cv2.imshow("Laser point", maskWar)
cv2.waitKey(0)
cv2.destroyAllWindows()

# STEP 5: Velocity calculation of laser point


# STEP 6: Show results
