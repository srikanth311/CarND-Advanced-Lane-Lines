import collections

class GlobalLaneVariables(object):
    global_left_lane_points_x = collections.deque(maxlen=10)
    global_left_lane_points_y = collections.deque(maxlen=10)
    global_right_lane_points_x = collections.deque(maxlen=10)
    global_right_lane_points_y = collections.deque(maxlen=10)
    global_left_lane_poly = None
    global_right_lane_poly = None