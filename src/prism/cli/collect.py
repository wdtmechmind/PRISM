import os
import sys


DEFAULT_MVS_ROOT = '/opt/MVS'
DEFAULT_LEGACY_RECORDING_DIR = '/opt/MVS/Samples/64/Python/General/Recording'
DEFAULT_MVIMPORT_DIR = '/opt/MVS/Samples/64/Python/MvImport'


def _prepend_path(path):
    if path and path not in sys.path:
        sys.path.insert(0, path)


def _truthy_env(name):
    return os.environ.get(name, '').strip().lower() in ['1', 'y', 'yes', 'true']


def _ensure_mvs_environment():
    if not os.environ.get('MVCAM_COMMON_RUNENV'):
        raise RuntimeError(
            'MVCAM_COMMON_RUNENV is not set. Run: source /opt/MVS/bin/set_env_path.sh /opt/MVS'
        )


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    _ensure_mvs_environment()
    mvimport_dir = os.environ.get('PRISM_MVIMPORT_DIR', DEFAULT_MVIMPORT_DIR)
    _prepend_path(mvimport_dir)

    use_legacy = _truthy_env('PRISM_LEGACY_COLLECT')
    if use_legacy:
        legacy_recording_dir = os.environ.get('PRISM_LEGACY_RECORDING_DIR', DEFAULT_LEGACY_RECORDING_DIR)
        _prepend_path(legacy_recording_dir)
        from DexHandDataCapture5CamMultiThread import main as collect_main
    else:
        from prism.online.session_manager import main as collect_main

    old_argv = sys.argv[:]
    try:
        sys.argv = ['prism-collect'] + list(argv)
        if use_legacy:
            return collect_main()
        return collect_main(list(argv))
    finally:
        sys.argv = old_argv


if __name__ == '__main__':
    main()
