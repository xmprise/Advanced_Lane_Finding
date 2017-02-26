import numpy as np
import cv2
import glob
import pickle


def get_camera_calibration(img):

    cal_pickle = pickle.load(open("camera_cal/camera_cal.p", "rb"))
    mtx = cal_pickle["mtx"]
    dist = cal_pickle["dist"]

    return cv2.undistort(img, mtx, dist, None, mtx)


def get_parameter():

    cal_pickle = pickle.load(open("camera_cal/camera_cal.p", "rb"))
    mtx = cal_pickle["mtx"]
    dist = cal_pickle["dist"]

    return mtx, dist


def calibration():
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    # all combination dimension
    objp1 = np.zeros((6 * 9, 3), np.float32)
    objp1[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    objp2 = np.zeros((6 * 8, 3), np.float32)
    objp2[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)
    objp3 = np.zeros((5 * 9, 3), np.float32)
    objp3[:, :2] = np.mgrid[0:9, 0:5].T.reshape(-1, 2)
    objp4 = np.zeros((4 * 9, 3), np.float32)
    objp4[:, :2] = np.mgrid[0:9, 0:4].T.reshape(-1, 2)
    objp5 = np.zeros((6 * 7, 3), np.float32)
    objp5[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
    objp6 = np.zeros((6 * 5, 3), np.float32)
    objp6[:, :2] = np.mgrid[0:5, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('camera_cal/calibration*.jpg')

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        img_size = (img.shape[1], img.shape[0])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        objp = objp1
        if not ret:
            ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)
            objp = objp2
        if not ret:
            ret, corners = cv2.findChessboardCorners(gray, (9, 5), None)
            objp = objp3
        if not ret:
            ret, corners = cv2.findChessboardCorners(gray, (9, 4), None)
            objp = objp4
        if not ret:
            ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)
            objp = objp5
        if not ret:
            ret, corners = cv2.findChessboardCorners(gray, (5, 6), None)
            objp = objp6

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Do camera calibration given object points and image points
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    # Save calibration result
    camera_cal_file = "camera_cal/camera_cal.p"
    output = open(camera_cal_file, 'wb')

    dict = {'mtx': 1, 'dist': 2}
    dict['mtx'] = mtx
    dict['dist'] = dist
    pickle.dump(dict, output)

    output.close()
