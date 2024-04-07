import cv2 as cv
import numpy as np
from cv2 import aruco
import argparse
from collections import defaultdict
from helpers import load_dict, load_calib_data, create_detector_params, get_marker_positions_from_base_marker

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

ID_TO_Z_AXIS = {
    0: (np.zeros((100,3)), 0),
    1: (np.zeros((100,3)), 0),
    2: (np.zeros((100,3)), 0),
    3: (np.zeros((100,3)), 0),
    4: (np.zeros((100,3)), 0),
    5: (np.zeros((100,3)), 0),
    6: (np.zeros((100,3)), 0),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get the position of the marker")
    parser.add_argument("-c", "--camera", type=int, help="Enter camera number")
    parser.add_argument("-n", "--name", type=str, help="Enter camera name")
    parser.add_argument("-m", "--marker_size", type=float, help="Enter the large marker size in centimeters")
    args = parser.parse_args()

    if args.camera is None:
        print("Please enter camera number using -c or --camera")
        exit()
    else:
        CAMERA_NUMBER = args.camera
        print(f"Using camera number {CAMERA_NUMBER}")

    if args.name is None:
        print("Please enter camera name using -n or --name")
        exit()
    else:
        CAMERA_NAME = args.name
        print(f"Using camera name {CAMERA_NAME}")

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
    cam_mat, dist_coef = load_calib_data(CAMERA_NAME)

    detector_params = create_detector_params()

    cap = cv.VideoCapture(CAMERA_NUMBER)

    last_base_marker_tvec = None
    last_base_marker_rvec = None

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
                    last_base_marker_tvec = tVec
                    last_base_marker_rvec = rVec
                    point = cv.drawFrameAxes(frame, cam_mat, dist_coef, last_base_marker_rvec, last_base_marker_tvec, marker_size, 4)

            if last_base_marker_tvec is not None and last_base_marker_rvec is not None and len(marker_IDs) >= 1:
                composedRvecs, composedTvecs = get_marker_positions_from_base_marker(last_base_marker_rvec, last_base_marker_tvec, marker_rvecs, marker_tvecs, debug=False)
                baseRvec = last_base_marker_rvec
                baseTvec = last_base_marker_tvec

                for marker_ID, composedRvec in composedRvecs.items():
                    composedTvec = composedTvecs[marker_ID]

                    info = cv.composeRT(composedRvec, composedTvec, baseRvec.T, baseTvec.T)
                    TcomposedRvec, TcomposedTvec = info[0], info[1]


                    z_axis = np.array([0,0,1])
                    rotV, _ = cv.Rodrigues(composedRvec)
                    new_z = np.around(rotV @ z_axis, 0)

                    values, count = ID_TO_Z_AXIS[marker_ID]
                    count += 1

                    new_values = np.copy(values)
                    new_values[count % 100] = new_z

                    ID_TO_Z_AXIS[marker_ID] = (new_values, count)


                    cv.drawFrameAxes(frame, cam_mat, dist_coef, TcomposedRvec, TcomposedTvec, 1.5, 4)

            for marker_ID in range(6):
                print(marker_ID, ": ", np.average(ID_TO_Z_AXIS[marker_ID][0], axis=0))


        cv.imshow("frame", frame)
        key = cv.waitKey(1)
        if key == ord("q"):
            break
        
    cap.release()
    cv.destroyAllWindows()
