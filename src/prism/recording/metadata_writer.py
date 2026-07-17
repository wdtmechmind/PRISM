import os


def resolve_output_root(output_arg, timestamp):
    session_name = 'DexHand_5Cam_%s' % timestamp
    if output_arg.strip():
        base = os.path.abspath(os.path.expanduser(output_arg.strip()))
        session_dir = os.path.join(base, session_name)
        os.makedirs(session_dir, exist_ok=True)
        return session_dir

    preferred = os.path.abspath(session_name)
    try:
        os.makedirs(preferred, exist_ok=True)
        return preferred
    except OSError:
        home_fallback = os.path.join(os.path.expanduser('~'), session_name)
        os.makedirs(home_fallback, exist_ok=True)
        print('warning: no write permission in current directory, fallback to: %s' % home_fallback)
        return home_fallback
