def target_position(start_position, relative_direction):
    return (start_position[0] + relative_direction[0],
            start_position[1] + relative_direction[1])


def hex_distance(a, b):
    return (abs(a[0] - b[0]) + abs(a[0] + a[1] - b[0] - b[1]) + abs(a[1] - b[1])) / 2