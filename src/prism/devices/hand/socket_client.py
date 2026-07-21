import socket
import time


GESTURE_TABLE = [
    (1, 'five_grasp', 'Five-finger grasp'),
    (2, 'five_open', 'Five-finger open'),
    (3, 'two_grasp_a', 'Two-finger grasp (A)'),
    (4, 'two_open_a', 'Two-finger open (A)'),
    (5, 'two_grasp_b', 'Two-finger grasp (B)'),
    (6, 'two_open_b', 'Two-finger open (B)'),
    (7, 'three_grasp_a', 'Three-finger grasp (A)'),
    (8, 'three_open_a', 'Three-finger open (A)'),
    (9, 'three_grasp_b', 'Three-finger grasp (B)'),
    (10, 'three_open_b', 'Three-finger open (B)'),
    (11, 'five_sequence', 'Five-finger sequence motion'),
    (12, 'thumb_in', 'Thumb inward'),
    (13, 'thumb_out', 'Thumb outward'),
    (14, 'index_point', 'Index pointing'),
    (15, 'index_press', 'Index press'),
    (16, 'index_single_click', 'Index single click'),
    (17, 'index_double_click', 'Index double click'),
]

GESTURE_ID_TO_POSE = {gid: pose for gid, pose, _ in GESTURE_TABLE}
GESTURE_ID_TO_NAME = {gid: name for gid, _, name in GESTURE_TABLE}

# Rule from DexHand protocol: gesture_id = ROG value + 1.
POSE_TO_COMMAND = {
    pose: '@ROG<%d>&' % (gesture_id - 1)
    for gesture_id, pose, _ in GESTURE_TABLE
}

POSE_ALIASES = {
    'grasp': 'five_grasp',
    'open': 'five_open',
    'three_grasp': 'three_grasp_a',
    'index_click': 'index_single_click',
}

POSE_TO_GESTURE_ID = {}
for gesture_id, pose_name in GESTURE_ID_TO_POSE.items():
    POSE_TO_GESTURE_ID[pose_name] = gesture_id
for alias, canonical in POSE_ALIASES.items():
    POSE_TO_GESTURE_ID[alias] = POSE_TO_GESTURE_ID[canonical]


class MechHandClient(object):
    def __init__(self, ip='127.0.0.1', port=60686, timeout_s=3.0, settle_time_s=1.0):
        self.ip = ip
        self.port = int(port)
        self.timeout_s = float(timeout_s)
        self.settle_time_s = max(0.0, float(settle_time_s))
        self.sock = None

    def connect(self):
        if self.sock is not None:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.timeout_s)
        sock.connect((self.ip, self.port))
        self.sock = sock

    def close(self):
        if self.sock is None:
            return
        try:
            self.sock.close()
        finally:
            self.sock = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def send_raw(self, command):
        if self.sock is None:
            raise RuntimeError('MechHandClient is not connected.')

        cmd = command.strip()
        if not cmd:
            raise ValueError('command cannot be empty')
        if not cmd.endswith('&'):
            cmd += '&'

        self.sock.sendall(cmd.encode('utf-8'))
        if self.settle_time_s > 0:
            time.sleep(self.settle_time_s)
        return cmd

    def send_pose(self, pose_name):
        canonical = POSE_ALIASES.get(pose_name, pose_name)
        if canonical not in POSE_TO_COMMAND:
            valid = ', '.join(sorted(POSE_TO_GESTURE_ID.keys()))
            raise ValueError('Unknown pose %r. Valid poses: %s' % (pose_name, valid))
        return self.send_raw(POSE_TO_COMMAND[canonical])

    def send_gesture(self, gesture_id):
        gid = int(gesture_id)
        if gid not in GESTURE_ID_TO_POSE:
            raise ValueError('Unknown gesture id %r. Valid range: 1-17' % gesture_id)
        return self.send_pose(GESTURE_ID_TO_POSE[gid])

    def grasp(self):
        return self.send_pose('five_grasp')

    def open_hand(self):
        return self.send_pose('five_open')

    def three_finger_grasp(self):
        return self.send_pose('three_grasp_a')

    def index_click(self):
        return self.send_pose('index_single_click')
