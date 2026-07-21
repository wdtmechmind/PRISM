import socket
import time


POSE_TO_COMMAND = {
    'grasp': '@ROG<0>&',
    'open': '@ROG<1>&',
    'three_grasp': '@ROG<6>&',
    'index_click': '@ROG<15>&',
}


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
        if pose_name not in POSE_TO_COMMAND:
            valid = ', '.join(sorted(POSE_TO_COMMAND.keys()))
            raise ValueError('Unknown pose %r. Valid poses: %s' % (pose_name, valid))
        return self.send_raw(POSE_TO_COMMAND[pose_name])

    def grasp(self):
        return self.send_pose('grasp')

    def open_hand(self):
        return self.send_pose('open')

    def three_finger_grasp(self):
        return self.send_pose('three_grasp')

    def index_click(self):
        return self.send_pose('index_click')
