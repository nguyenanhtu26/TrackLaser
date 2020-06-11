from laserFunctions import *
for i in range(1, 403):
    print(i)
    filename = 'images/video6-7/anhOut' + str(i) + '.jpg'
    image = cv2.imread(filename)
    # [x, y] = FindCenterLaserPointer(image)
    [x, y] = FindCenter(image)
    [verCoor, honCoor] = GridCoordinates(image)
    [x_real, y_real, img] = RealCoordinatesOfLaserPointer(image, x, y, verCoor, honCoor)
    cv2.imshow("Original Image", image)
    cv2.imwrite('detected Image/video6-7-2/' + str(i) + '.jpg', img)
    print(x_real, y_real)
cv2.waitKey(0)
cv2.destroyAllWindows()