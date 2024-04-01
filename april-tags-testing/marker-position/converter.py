def convertToRobotMoves(solution):
    face_to_rotation_map = {
        'U': 'x x',
        'D': '',
        'F': 'x x x',
        'L': 'z z z',
        'B': 'x',
        'R': 'z',
    }

    move_affect_map = {
        'U1': face_to_rotation_map['U'] + ' y',
        'D1': face_to_rotation_map['D'] + ' y',
        'F1': face_to_rotation_map['F'] + ' y',
        'L1': face_to_rotation_map['L'] + ' y',
        'B1': face_to_rotation_map['B'] + ' y',
        'R1': face_to_rotation_map['R'] + ' y',

        'U2': face_to_rotation_map['U'] + ' y y',
        'D2': face_to_rotation_map['D'] + ' y y',
        'F2': face_to_rotation_map['F'] + ' y y',
        'L2': face_to_rotation_map['L'] + ' y y',
        'B2': face_to_rotation_map['B'] + ' y y',
        'R2': face_to_rotation_map['R'] + ' y y',

        'U3': face_to_rotation_map['U'] + ' y y y',
        'D3': face_to_rotation_map['D'] + ' y y y',
        'F3': face_to_rotation_map['F'] + ' y y y',
        'L3': face_to_rotation_map['L'] + ' y y y',
        'B3': face_to_rotation_map['B'] + ' y y y',
        'R3': face_to_rotation_map['R'] + ' y y y',
    }

    z_map = {
        'F': 'F',
        'B': 'B',
        'U': 'R',
        'R': 'D',
        'D': 'L',
        'L': 'U'
    }

    x_map = {
        'L': 'L',
        'R': 'R',
        'U': 'B',
        'B': 'D',
        'D': 'F',
        'F': 'U'
    }

    y_map = {
        'U': 'U',
        'D': 'D',
        'F': 'L',
        'L': 'B',
        'B': 'R',
        'R': 'F'
    }

    readable_converter = {
        'U': 'x2',
        'D': '',
        'F': "x3",
        'L': "z3",
        'B': 'x1',
        'R': 'z1',
    }

    moves = solution.split(' ')
    new_solution = []

    for i in range(len(moves)):
        current_move = moves[i]
        direction = current_move[1]
        new_solution.append(readable_converter[current_move[0]])
        new_solution.append(f'Uw{direction}')

        rotations = move_affect_map[current_move].split(' ')

        for j in range(i + 1, len(moves)):
            direction = moves[j][1]

            for rotation in rotations:
                if rotation == 'x':
                    moves[j] = x_map[moves[j][0]]
                elif rotation == 'y':
                    moves[j] = y_map[moves[j][0]]
                elif rotation == 'z':
                    moves[j] = z_map[moves[j][0]]
                moves[j] += direction
            
    print(' '.join(new_solution))

