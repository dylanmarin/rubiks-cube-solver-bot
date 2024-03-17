import cv2 as cv
import numpy as np
from cv2 import aruco
import argparse
from collections import defaultdict

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

def draw_cube(img, R, t, camera_mat, dist_coef, color, sidelength = 5.6):
    '''
    Referenced from: 
        How to draw 3D Coordinate Axes with OpenCV for face pose estimation?
        https://stackoverflow.com/questions/30207467/how-to-draw-3d-coordinate-axes-with-opencv-for-face-pose-estimation
    '''
    # unit is mm
    rotV, _ = cv.Rodrigues(R)
    points = np.float32([[-sidelength/2, sidelength/2, 0],  [-sidelength/2,-sidelength/2,0],[sidelength/2, -sidelength/2, 0], [sidelength/2, sidelength/2, 0], [-sidelength/2, sidelength/2, -sidelength],  [-sidelength/2,-sidelength/2, -sidelength],[sidelength/2, -sidelength/2, -sidelength], [sidelength/2, sidelength/2, -sidelength]]).reshape(-1, 3)
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


def inversePerspective(rvec, tvec):
    ''' 
    Get the inverse perspective of the rvec and tvec
    Referenced from:
    https://aliyasineser.medium.com/calculation-relative-positions-of-aruco-markers-eee9cc4036e3
    '''
    R, _ = cv.Rodrigues(rvec)
    R = np.matrix(R).T
    invTvec = np.dot(R, np.matrix(-tvec))
    invRvec, _ = cv.Rodrigues(R)
    return invRvec, invTvec


def relativePosition(rvec1, tvec1, rvec2, tvec2):
    """ 
    Get relative position for rvec2 & tvec2. Compose the returned rvec & tvec to use composeRT with rvec2 & tvec2 
    Referenced from:
    https://aliyasineser.medium.com/calculation-relative-positions-of-aruco-markers-eee9cc4036e3
    """
    rvec1, tvec1 = rvec1.reshape((3, 1)), tvec1.reshape((3, 1))
    rvec2, tvec2 = rvec2.reshape((3, 1)), tvec2.reshape((3, 1))
    # Inverse the second marker
    invRvec, invTvec = inversePerspective(rvec2, tvec2)
    info = cv.composeRT(rvec1, tvec1, invRvec, invTvec)
    composedRvec, composedTvec = info[0], info[1]
    composedRvec = composedRvec.reshape((3, 1))
    composedTvec = composedTvec.reshape((3, 1))
    return composedRvec, composedTvec


def get_marker_positions_from_base_marker(marker_rvecs, marker_tvecs, debug=True):
    '''
    Get the position of the markers from the base marker
    '''
    base_marker_id = 9
    base_marker_rvec = marker_rvecs[base_marker_id]
    base_marker_tvec = marker_tvecs[base_marker_id]

    output_rvecs = defaultdict()
    output_tvecs = defaultdict()

    for marker_id, rvec in marker_rvecs.items():
        if marker_id != base_marker_id:
            tvec = marker_tvecs[marker_id]
            composedRvec, composedTvec = relativePosition(rvec, tvec,base_marker_rvec, base_marker_tvec)
            output_rvecs[marker_id] = composedRvec.ravel()
            output_tvecs[marker_id] = composedTvec.ravel()
            if debug:
                print(f"Marker {marker_id} position: ")
                print(f"x: {composedTvec[0][0]}")
                print(f"y: {composedTvec[1][0]}")
                print(f"z: {composedTvec[2][0]}")

    return output_rvecs, output_tvecs

ID_TO_COLOR = {
    0: (0, 255, 0),
    1: (0, 165, 255),
    2: (0, 255, 255),
    3: (0, 0, 255),
    4: (255, 0, 0),
    5: (255, 255, 255),
    6: (255, 100, 255),
    7: (255, 100, 255),
    8: (255, 100, 255),
    9: (255, 100, 255)
}




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get the position of the marker")
    parser.add_argument("-c", "--camera", type=int, help="Enter camera number")
    parser.add_argument("-m", "--marker_size", type=float, help="Enter the large marker size in centimeters")
    args = parser.parse_args()

    if args.camera is None:
        print("Please enter camera number using -c or --camera")
        exit()
    else:
        CAMERA_NUMBER = args.camera
        print(f"Using camera number {CAMERA_NUMBER}")

    if args.marker_size is None:
        LARGE_MARKER_SIZE = 3 # centimeters
        print(f"Using default large marker size {LARGE_MARKER_SIZE}")
    else:
        LARGE_MARKER_SIZE = args.marker_size
        print(f"Using marker size {LARGE_MARKER_SIZE}")
    
    def get_marker_size(marker_id):
        '''
        Get the marker size in centimeters
        '''
        if marker_id < 6:
            return 1.5
        else:
            return LARGE_MARKER_SIZE

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
            
            marker_rvecs = defaultdict()
            marker_tvecs = defaultdict()

            for i in range(len(marker_IDs)):
                marker_ID = marker_IDs[i][0]
                marker_size = get_marker_size(marker_ID)
                rVec, tVec, _ = aruco.estimatePoseSingleMarkers(
                    marker_corners[i], marker_size, cam_mat, dist_coef
                )
                marker_rvecs[marker_ID] = rVec[0][0]
                marker_tvecs[marker_ID] = tVec[0][0]

            for marker_ID, rVec in marker_rvecs.items():
                tVec = marker_tvecs[marker_ID]
                marker_size = get_marker_size(marker_ID)
                # if marker_ID < 6:
                #     # frame = draw_cube(frame, rVec, tVec, cam_mat, dist_coef, ID_TO_COLOR[marker_ID], sidelength=5.5)
                #     point = cv.drawFrameAxes(frame, cam_mat, dist_coef, rVec, tVec, marker_size, 4)
                # else:
                #     # frame = draw_cube(frame, rVec, tVec, cam_mat, dist_coef, ID_TO_COLOR[marker_ID], sidelength=marker_size)
                #     point = cv.drawFrameAxes(frame, cam_mat, dist_coef, rVec, tVec, marker_size, 4)
                if marker_ID == 9:
                    point = cv.drawFrameAxes(frame, cam_mat, dist_coef, rVec, tVec, marker_size, 4)

            if [9] in marker_IDs and len(marker_IDs) > 1:
                composedRvecs, composedTvecs = get_marker_positions_from_base_marker(marker_rvecs, marker_tvecs, debug=True)
                baseRvec = marker_rvecs[9]
                baseTvec = marker_tvecs[9]

                for marker_ID, composedRvec in composedRvecs.items():
                    composedTvec = composedTvecs[marker_ID]

                    info = cv.composeRT(composedRvec, composedTvec, baseRvec.T, baseTvec.T)
                    TcomposedRvec, TcomposedTvec = info[0], info[1]

                    # print(f"Marker {marker_ID} position: ")
                    # print(f"x: {TcomposedTvec[0]} --- {marker_tvecs[marker_ID][0]}")
                    # print(f"y: {TcomposedTvec[1]} --- {marker_tvecs[marker_ID][1]}")
                    # print(f"z: {TcomposedTvec[2]} --- {marker_tvecs[marker_ID][2]}")

                    cv.drawFrameAxes(frame, cam_mat, dist_coef, TcomposedRvec, TcomposedTvec, 1.5, 4)


        cv.imshow("frame", frame)
        key = cv.waitKey(1)
        if key == ord("q"):
            break
        
    cap.release()
    cv.destroyAllWindows()
