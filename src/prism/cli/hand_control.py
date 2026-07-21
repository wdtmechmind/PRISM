import argparse
import sys

from prism.devices.hand import MechHandClient


MENU_TEXT = """
========= MechHand =========

1 Five-finger grasp
2 Five-finger open
3 Three-finger grasp
4 Index click
0 Exit
""".strip('\n')


KEY_TO_POSE = {
    '1': 'grasp',
    '2': 'open',
    '3': 'three_grasp',
    '4': 'index_click',
}


def build_parser():
    parser = argparse.ArgumentParser(
        description='Control mech hand poses through socket commands.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--ip', type=str, default='127.0.0.1', help='Mech hand controller IP')
    parser.add_argument('--port', type=int, default=60686, help='Mech hand controller TCP port')
    parser.add_argument('--timeout-s', type=float, default=3.0, help='Socket connect/send timeout')
    parser.add_argument('--settle-time-s', type=float, default=1.0,
                        help='Delay after each command so the hand can move')
    parser.add_argument('--pose', type=str, default='', choices=['', 'grasp', 'open', 'three_grasp', 'index_click'],
                        help='One-shot pose command. Leave empty to enter interactive mode')
    parser.add_argument('--raw-cmd', type=str, default='',
                        help='One-shot raw command, e.g. @ROG<0>&. When set, it overrides --pose')
    return parser


def run_interactive(client):
    while True:
        print('\n' + MENU_TEXT + '\n')
        key = input('> ').strip()
        if key == '0':
            return 0
        if key not in KEY_TO_POSE:
            print('Unknown input. Please choose 0/1/2/3/4.')
            continue
        pose = KEY_TO_POSE[key]
        sent = client.send_pose(pose)
        print('Sent:', sent)


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        with MechHandClient(
            ip=args.ip,
            port=args.port,
            timeout_s=args.timeout_s,
            settle_time_s=args.settle_time_s,
        ) as client:
            print('MechHand connected to %s:%s' % (args.ip, args.port))

            if args.raw_cmd:
                sent = client.send_raw(args.raw_cmd)
                print('Sent:', sent)
                return 0

            if args.pose:
                sent = client.send_pose(args.pose)
                print('Sent:', sent)
                return 0

            return run_interactive(client)
    except KeyboardInterrupt:
        print('\nInterrupted.')
        return 130
    except Exception as exc:
        print('Error:', str(exc))
        return 1


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
