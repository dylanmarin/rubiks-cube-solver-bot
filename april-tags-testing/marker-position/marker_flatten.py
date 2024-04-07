import cv2 as cv
import numpy as np
from cv2 import aruco
import argparse
from collections import defaultdict
import twophase.solver as sv
from helpers import load_dict, load_calib_data, create_detector_params, get_marker_size

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


colors = {
    'w': (225, 225, 225), # white
    'y': (75, 235, 220), # yellow
    'g': (60, 230, 110), # green
    'b': (225, 120, 20), # blue
    'o': (50, 100, 255), # orange
    'r': (70, 70, 240), # red
}

marker_to_color = {
    0: 'w',
    1: 'r',
    2: 'b',
    3: 'g',
    4: 'o',
    5: 'y'
}

color_to_marker = {
    'w': 0,
    'r': 1,
    'b': 2,
    'g': 3,
    'o': 4,
    'y': 5
}


color_to_side = {
    'w': 'U',
    'y': 'D',
    'g': 'F',
    'b': 'B',
    'r': 'R',
    'o': 'L'
}

def normalize(v):
    return v / np.linalg.norm(v, ord=2)

def get_color(r,g,b): # compare rgb values and return color
    min_distance = np.inf
    best_color = None

    input = normalize(np.array((r, g, b)))

    for (key, color) in colors.items():
        dist = np.linalg.norm(input - normalize(np.array(color)), ord=1)
        if dist < min_distance:
            min_distance = dist
            best_color = key, color
    return best_color

def convert_to_string(cube_array):
    output = ''

    order = [0, 1, 3, 5, 4, 2]

    for k in range(6):
        for i in range(3):
            for j in range(3):
                output += color_to_side[marker_to_color[cube_array[order[k]][j][i]]]

    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get the position of the marker")
    parser.add_argument("-c", "--camera", type=int, help="Enter camera number")
    parser.add_argument("-n", "--name", type=str, help="Enter camera name")

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

    marker_dict = load_dict()
    cam_mat, dist_coef = load_calib_data(CAMERA_NAME)

    detector_params = create_detector_params()

    cap = cv.VideoCapture(CAMERA_NUMBER)

    cube = np.full((6, 3, 3), -1)
    prevs = np.full((30, 6, 3, 3), -1)
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        marker_corners, marker_IDs, reject = aruco.detectMarkers(
            gray_frame, marker_dict, parameters=detector_params
        )
        current_marker = None
        
        if len(marker_corners) > 0:
            dists = [aruco.estimatePoseSingleMarkers(c, 1.5, cam_mat, dist_coef)[1][0][0][2] for c in marker_corners]

            marker_index = np.argmin(dists)
            if 0 <= marker_index < 6:
                current_marker = marker_IDs[marker_index][0]

        if marker_corners:
            for corners, marker_id in zip(marker_corners, marker_IDs):
                # Needed to ignore marker orientation
                smallest_point_index = np.argmin(np.sum(corners[0], axis=1))
                corners[0] = np.roll(corners[0], 3 - smallest_point_index, axis=0)

                transform = cv.getPerspectiveTransform(np.float32(corners), np.array([[200, 200], [200, 300], [300, 300], [300, 200]], dtype=np.float32))
                frame = cv.warpPerspective(frame, transform,(500, 500),flags=cv.INTER_LINEAR)
                break

        center = (250, 250)
        square_size = 60
        half_square = int(square_size / 2)
        gap = 50

        frame = cv.flip(frame, 1)

        frame2 = frame.copy()
        display = np.zeros((500, 500, 3))

        for j in range(3):
            startY = center[1] - half_square - gap - square_size + j * (gap + square_size)
            endY = startY + square_size

            for i in range(3):
                startX = center[0] - half_square - gap - square_size + i * (gap + square_size)
                endX = startX + square_size

                try:
                    cv.rectangle(frame, (startX, startY), (endX, endY), (255,255,255), 3)

                    if i == j == 1:
                        display[startY:endY, startX:endX] = colors[marker_to_color[current_marker]]
                        cube[current_marker][i][j] = current_marker
                        continue

                    if current_marker is None: continue

                    subframe = frame2[startY:endY, startX:endX]

                    r, g, b = cv.split(subframe)
                    r_avg = int(cv.mean(r)[0])
                    g_avg = int(cv.mean(g)[0])
                    b_avg = int(cv.mean(b)[0])
                    # print(r_avg, g_avg, b_avg)
                    key, color = get_color(r_avg, g_avg, b_avg)

                    marker = color_to_marker[key]
                    prevs[count][current_marker][i][j] = marker

                    if np.all(prevs[:, current_marker, i, j] == marker):
                        display[startY:endY, startX:endX] = color
                        cube[current_marker][i][j] = marker

                except:
                    continue

        cv.imshow("frame", frame)
        cv.imshow("display", display / 255)
        key = cv.waitKey(1)
        if key == ord("q"):
            break

        count += 1
        count %= prevs.shape[0]

    cube_string = convert_to_string(cube)
    print(cube_string[:9])
    print(cube_string[9: 18])
    print(cube_string[18: 27])
    print(cube_string[27: 36])
    print(cube_string[36: 45])
    print(cube_string[45: 54])

    print(sv.solve(cube_string))
    cap.release()
    cv.destroyAllWindows()


