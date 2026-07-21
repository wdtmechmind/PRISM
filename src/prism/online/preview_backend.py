import collections
import os

import cv2
import numpy as np


class OpenCVPreviewBackend(object):
    def __init__(self, window_name, width, height, mouse_callback=None):
        self.window_name = window_name
        self.width = int(width)
        self.height = int(height)
        self.mouse_callback = mouse_callback

    def open(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, max(640, self.width), max(480, self.height))
        if self.mouse_callback is not None:
            cv2.setMouseCallback(self.window_name, self.mouse_callback)

    def show_frame(self, bgr_frame):
        cv2.imshow(self.window_name, bgr_frame)

    def poll_key(self):
        return cv2.waitKey(1) & 0xFF

    def get_window_size(self):
        # OpenCV does not always expose reliable live client size cross-platform.
        return max(640, self.width), max(480, self.height)

    def close(self):
        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass


class QtPreviewBackend(object):
    def __init__(self, window_name, width, height):
        self.window_name = window_name
        self.width = int(width)
        self.height = int(height)
        self._queue = collections.deque()
        self._closed = False
        self._app = None
        self._window = None
        self._label = None
        self._qt = None

    def open(self):
        self._sanitize_qt_env_for_cv2_conflict()
        try:
            from PySide6 import QtCore, QtGui, QtWidgets
        except Exception:
            try:
                from PyQt5 import QtCore, QtGui, QtWidgets
            except Exception as ex:
                raise RuntimeError(
                    'Qt backend requested but neither PySide6 nor PyQt5 is available. '
                    'Install one of them and re-run with --ui-backend qt.'
                ) from ex

        self._qt = (QtCore, QtGui, QtWidgets)
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])
        self._app = app

        backend = self

        class _PreviewWindow(QtWidgets.QWidget):
            def __init__(self):
                super(_PreviewWindow, self).__init__()
                self.setWindowTitle(backend.window_name)
                self.setFocusPolicy(QtCore.Qt.StrongFocus)
                self.resize(max(640, backend.width), max(480, backend.height))

                layout = QtWidgets.QVBoxLayout(self)
                layout.setContentsMargins(0, 0, 0, 0)
                label = QtWidgets.QLabel(self)
                label.setAlignment(QtCore.Qt.AlignCenter)
                label.setMinimumSize(320, 240)
                layout.addWidget(label)
                backend._label = label

            def keyPressEvent(self, event):
                code = _qt_key_to_cv_code(event)
                if code is not None:
                    backend._queue.append(code)
                super(_PreviewWindow, self).keyPressEvent(event)

            def closeEvent(self, event):
                backend._closed = True
                super(_PreviewWindow, self).closeEvent(event)

        def _qt_key_to_cv_code(event):
            key = event.key()
            text = event.text() or ''
            if text:
                ch = text[0]
                if '0' <= ch <= '9':
                    return ord(ch)
                if 'a' <= ch.lower() <= 'z':
                    return ord(ch.lower())
                if ch == ' ':
                    return ord(' ')

            if key == QtCore.Qt.Key_Escape:
                return 27
            if key in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
                return 13
            if key == QtCore.Qt.Key_Backspace:
                return 8
            if key == QtCore.Qt.Key_Space:
                return ord(' ')
            return None

        self._window = _PreviewWindow()
        self._window.show()
        self._window.activateWindow()
        self._window.setFocus()
        self._app.processEvents()

    @staticmethod
    def _sanitize_qt_env_for_cv2_conflict():
        # OpenCV wheels often set Qt plugin env vars to cv2/qt/plugins, which can
        # conflict with PySide/PyQt runtime loading and cause xcb plugin aborts.
        keys = ['QT_PLUGIN_PATH', 'QT_QPA_PLATFORM_PLUGIN_PATH']
        for key in keys:
            val = os.environ.get(key, '')
            low = val.lower()
            if 'cv2' in low and 'qt' in low and 'plugin' in low:
                os.environ.pop(key, None)

    def show_frame(self, bgr_frame):
        QtCore, QtGui, _QtWidgets = self._qt
        if self._window is None or self._label is None:
            return

        if self._closed:
            return

        # Avoid cv2 usage in Qt backend path to reduce Qt plugin interaction surface.
        rgb = np.ascontiguousarray(bgr_frame[:, :, ::-1])
        h, w = rgb.shape[:2]
        qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()
        pix = QtGui.QPixmap.fromImage(qimg)
        target = self._label.size()
        if target.width() > 0 and target.height() > 0:
            pix = pix.scaled(target, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self._label.setPixmap(pix)
        self._app.processEvents()

    def poll_key(self):
        if self._app is not None:
            self._app.processEvents()
        if self._closed:
            return 27
        if self._queue:
            return self._queue.popleft()
        return -1

    def get_window_size(self):
        if self._window is None:
            return max(640, self.width), max(480, self.height)
        return max(640, self._window.width()), max(480, self._window.height())

    def close(self):
        if self._window is not None:
            try:
                self._window.close()
            except Exception:
                pass
            self._window = None
        self._label = None
        self._closed = True
        if self._app is not None:
            try:
                self._app.processEvents()
            except Exception:
                pass


def create_preview_backend(backend_name, window_name, width, height, mouse_callback=None):
    name = (backend_name or 'opencv').strip().lower()
    if name == 'opencv':
        return OpenCVPreviewBackend(window_name, width, height, mouse_callback=mouse_callback)
    if name == 'qt':
        # Qt backend currently keeps keyboard/CLI control only; no mouse callback wiring.
        return QtPreviewBackend(window_name, width, height)
    raise ValueError('Unknown ui backend %r. Use one of: opencv, qt' % backend_name)
