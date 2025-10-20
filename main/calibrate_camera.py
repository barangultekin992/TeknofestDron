import numpy as np
import cv2
import glob
import os

chessboard_size = (7,7)
resulution = (1920, 1080)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

objPoints = [] #3D dünya koordinat sisteminden noktalar 
imgPoints = [] #2D görsel koordinat sisteminden noktalar

images = glob.glob("/home/baran/ardu_ws/src/main/*.jpg")
for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"Failed to load image: {fname}")
        continue
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        print(f"Chessboard found in {fname}")
        objPoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgPoints.append(corners2)

        cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)
    else:
        print(f"Chessboard NOT found in {fname}")

ret, cameraMatrix, dist,rvecs,tvecs = cv2.calibrateCamera(objPoints, imgPoints, resulution, None, None)
print("Camera matrix:")
print(cameraMatrix)
print("Distortion coefficients:")
print(dist)
print("Rotation vectors:")
print(rvecs)
print("Translation vectors:")
print(tvecs)

np.save("camera_intrinsic.txt",cameraMatrix)
np.save("camera_intrinsic.txt",dist)
np.save("camera_intrinsic.txt",rvecs)
np.save("camera_intrinsic.txt",tvecs)

cv2.destroyAllWindows()
