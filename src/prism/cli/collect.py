import os
import sys


DEFAULT_MVS_ROOT = '/opt/MVS'
DEFAULT_LEGACY_RECORDING_DIR = '/opt/MVS/Samples/64/Python/General/Recording'
DEFAULT_MVIMPORT_DIR = '/opt/MVS/Samples/64/Python/MvImport'


def _prepend_path(path):
    if path and path not in sys.path:
        sys.path.insert(0, path)


def _ensure_mvs_environment():
    if not os.environ.get('MVCAM_COMMON_RUNENV'):
        raise RuntimeError(
            'MVCAM_COMMON_RUNENV is not set. Run: source /opt/MVS/bin/set_env_path.sh /opt/MVS'
        )


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    _ensure_mvs_environment()
    legacy_recording_dir = os.environ.get('PRISM_LEGACY_RECORDING_DIR', DEFAULT_LEGACY_RECORDING_DIR)
    mvimport_dir = os.environ.get('PRISM_MVIMPORT_DIR', DEFAULT_MVIMPORT_DIR)
    _prepend_path(mvimport_dir)
    _prepend_path(legacy_recording_dir)

    use_native = os.environ.get('PRISM_NATIVE_COLLECT', '').strip().lower() in ['1', 'y', 'yes', 'true']
    if use_native:
        from prism.online.session_manager import main as collect_main
    else:
        from DexHandDataCapture5CamMultiThread import main as collect_main

    old_argv = sys.argv[:]
    try:
        sys.argv = ['prism-collect'] + list(argv)
        if use_native:
            return collect_main(list(argv))
        return collect_main()
    finally:
        sys.argv = old_argv


if __name__ == '__main__':
    main()
