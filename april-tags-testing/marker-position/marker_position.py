import cv2 as cv
import numpy as np
from cv2 import aruco
import argparse

def load_dict():
    data = np.load('../markers-dict/dictionary.npz')
    loaded_dict = cv.aruco.Dictionary_create(10,3)
    loaded_dict.bytesList = data['bytesList']
    return loaded_dict

def load_calib_data(camera_number):
    calib_data_path = f"../calib-data/MultiMatrix_Camera_{camera_number}.npz"
    calib_data = np.load(calib_data_path)
    cam_mat = calib_data["camMatrix"]
    dist_coef = calib_data["distCoef"]
    return cam_mat, dist_coef

def create_detector_params():
    detector_params = aruco.DetectorParameters_create()
    # adaptiveThreshConstant = 7
    # adaptiveThreshWinSizeMin = 3
    # adaptiveThreshWinSizeMax = 23
    # adaptiveThreshWinSizeStep = 10
    return detector_params

def draw_cube(img, R, t, camera_mat, dist_coef, color):
    '''
    Referenced from: 
        How to draw 3D Coordinate Axes with OpenCV for face pose estimation?
        https://stackoverflow.com/questions/30207467/how-to-draw-3d-coordinate-axes-with-opencv-for-face-pose-estimation
    '''
    # unit is mm
    rotV, _ = cv.Rodrigues(R)
    points = np.float32([[-2.8, 2.8, 0],  [-2.8,-2.8,0],[2.8, -2.8, 0], [2.8, 2.8, 0], [-2.8, 2.8, -5.5],  [-2.8,-2.8, -5.5],[2.8, -2.8, -5.5], [2.8, 2.8, -5.5]]).reshape(-1, 3)
    axisPoints, _ = cv.projectPoints(points, rotV, t, camera_mat, dist_coef)

    # draw cube edges onto screen
    img = cv.line(img, tuple(axisPoints[0].ravel().astype(int)), tuple(axisPoints[1].ravel().astype(int)), color, 3)    
    img = cv.line(img, tuple(axisPoints[1].ravel().astype(int)), tuple(axisPoints[2].ravel().astype(int)), color, 3)
    img = cv.line(img, tuple(axisPoints[2].ravel().astype(int)), tuple(axisPoints[3].ravel().astype(int)), color, 3)    
    img = cv.line(img, tuple(axisPoints[3].ravel().astype(int)), tuple(axisPoints[0].ravel().astype(int)), color, 3)

    img = cv.line(img, tuple(axisPoints[4].ravel().astype(int)), tuple(axisPoints[5].ravel().astype(int)), color, 3)    
    img = cv.line(img, tuple(axisPoints[5].ravel().astype(int)), tuple(axisPoints[6].ravel().astype(int)), color, 3)
    img = cv.line(img, tuple(axisPoints[6].ravel().astype(int)), tuple(axisPoints[7].ravel().astype(int)), color, 3)    
    img = cv.line(img, tuple(axisPoints[7].ravel().astype(int)), tuple(axisPoints[4].ravel().astype(int)), color, 3)
 
    img = cv.line(img, tuple(axisPoints[0].ravel().astype(int)), tuple(axisPoints[4].ravel().astype(int)), color, 3)    
    img = cv.line(img, tuple(axisPoints[1].ravel().astype(int)), tuple(axisPoints[5].ravel().astype(int)), color, 3)
    img = cv.line(img, tuple(axisPoints[2].ravel().astype(int)), tuple(axisPoints[6].ravel().astype(int)), color, 3)    
    img = cv.line(img, tuple(axisPoints[3].ravel().astype(int)), tuple(axisPoints[7].ravel().astype(int)), color, 3)
    
    return img


ID_TO_COLOR_STRING = {
    0: "green",
    1: "orange",
    2: "yellow",
    3: "red",
    4: "blue",
    5: "white"
}

ID_TO_COLOR = {
    0: (0, 255, 0),
    1: (0, 165, 255),
    2: (0, 255, 255),
    3: (0, 0, 255),
    4: (255, 0, 0),
    5: (255, 255, 255)
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get the position of the marker")
    parser.add_argument("-c", "--camera", type=int, help="Enter camera number")
    parser.add_argument("-m", "--marker_size", type=float, help="Enter marker size in centimeters")
    args = parser.parse_args()

    if args.camera is None:
        print("Please enter camera number using -c or --camera")
        exit()
    else:
        CAMERA_NUMBER = args.camera
        print(f"Using camera number {CAMERA_NUMBER}")

    if args.marker_size is None:
        MARKER_SIZE = 1.5 # centimeters
        print(f"Using default marker size {MARKER_SIZE}")
    else:
        MARKER_SIZE = args.marker_size
        print(f"Using marker size {MARKER_SIZE}")
    
    marker_dict = load_dict()
    cam_mat, dist_coef = load_calib_data(CAMERA_NUMBER)

    detector_params = create_detector_params()

    cap = cv.VideoCapture(CAMERA_NUMBER)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        marker_corners, marker_IDs, reject = aruco.detectMarkers(
            gray_frame, marker_dict, parameters=detector_params
        )

        if marker_corners:
            # from the docs:
            # "The camera pose relative to the marker is a 3d transformation from the marker coordinate system to the camera coordinate system."
            rVec, tVec, _ = aruco.estimatePoseSingleMarkers(
                marker_corners, MARKER_SIZE, cam_mat, dist_coef
            )
            total_markers = range(0, marker_IDs.size)

            aruco.drawDetectedMarkers(frame, marker_corners, marker_IDs)

            for ids, corners, i in zip(marker_IDs, marker_corners, total_markers):
                cv.polylines(
                    frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv.LINE_AA
                )

                distance = np.sqrt(
                    tVec[i][0][2] ** 2 + tVec[i][0][0] ** 2 + tVec[i][0][1] ** 2
                )

                # Draw the cube estimated from each marker
                if ids[0] < 6:
                    frame = draw_cube(frame, rVec[i], tVec[i], cam_mat, dist_coef, ID_TO_COLOR[ids[0]])

                else:
                    point = cv.drawFrameAxes(frame, cam_mat, dist_coef, rVec[i], tVec[i], 2.9, 4)

                '''
                # cv.putText(
                #     frame,
                #     f"id: {ids[0]} Dist: {round(distance, 2)}",
                #     top_right,
                #     cv.FONT_HERSHEY_PLAIN,
                #     1.3,
                #     (0, 0, 255),
                #     2,
                #     cv.LINE_AA,
                # )
                # cv.putText(
                #     frame,
                #     f"x:{round(tVec[i][0][0],1)} y: {round(tVec[i][0][1],1)} ",
                #     bottom_right,
                #     cv.FONT_HERSHEY_PLAIN,
                #     1.0,
                #     (0, 0, 255),
                #     2,
                #     cv.LINE_AA,
                # )
                # print(ids, "  ", corners)
                '''


        cv.imshow("frame", frame)
        key = cv.waitKey(1)
        if key == ord("q"):
            break
        
    cap.release()
    cv.destroyAllWindows()
