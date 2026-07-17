from ctypes import *

import numpy as np

from MvCameraControl_class import *  # noqa: F401,F403


def decoding_char(ctypes_char_array):
    byte_str = memoryview(ctypes_char_array).tobytes()
    null_index = byte_str.find(b'\x00')
    if null_index != -1:
        byte_str = byte_str[:null_index]

    for encoding in ['utf-8', 'gbk', 'latin-1']:
        try:
            return byte_str.decode(encoding)
        except UnicodeDecodeError:
            continue
    return byte_str.decode('latin-1', errors='replace')


def enumerate_usb_devices():
    device_list = MV_CC_DEVICE_INFO_LIST()
    ret = MvCamera.MV_CC_EnumDevices(MV_USB_DEVICE, device_list)
    if ret != 0:
        raise Exception('enum usb devices fail[0x%x]' % ret)

    usb_infos = []
    for i in range(device_list.nDeviceNum):
        dev_info = cast(device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        if dev_info.nTLayerType != MV_USB_DEVICE:
            continue

        model = decoding_char(dev_info.SpecialInfo.stUsb3VInfo.chModelName)
        serial = decoding_char(dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber)
        usb_infos.append((i, dev_info, model, serial))

    return usb_infos


def parse_indices(raw_text, max_count, expected_count):
    tokens = [x.strip() for x in raw_text.split(',') if x.strip()]
    if len(tokens) != expected_count:
        raise ValueError('please input exactly %d indices, like: 0,1,2,3' % expected_count)

    indices = [int(x) for x in tokens]
    if len(set(indices)) != expected_count:
        raise ValueError('indices must be unique')

    for idx in indices:
        if idx < 0 or idx >= max_count:
            raise ValueError('index out of range: %d' % idx)

    return indices


class UsbCameraGrabber(object):
    def __init__(self, device_info, serial_number, model_name):
        self.device_info = device_info
        self.serial_number = serial_number
        self.model_name = model_name
        self.cam = MvCamera()
        self.started_grab = False

    def open_and_prepare(self, use_hardware_trigger, use_software_trigger, exposure_us=None, gain=None, frame_rate=None):
        ret = self.cam.MV_CC_CreateHandle(self.device_info)
        if ret != 0:
            raise Exception('create handle fail[0x%x], serial=%s' % (ret, self.serial_number))

        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            raise Exception('open device fail[0x%x], serial=%s' % (ret, self.serial_number))

        self.disable_auto_features()
        self.apply_manual_params(exposure_us=exposure_us, gain=gain, frame_rate=frame_rate)

        if use_hardware_trigger:
            ret = self.cam.MV_CC_SetEnumValueByString('TriggerMode', 'On')
            if ret != 0:
                raise Exception('set trigger mode on fail[0x%x], serial=%s' % (ret, self.serial_number))

            ret = self.cam.MV_CC_SetEnumValueByString('TriggerSource', 'Line0')
            if ret != 0:
                raise Exception('set trigger source line0 fail[0x%x], serial=%s' % (ret, self.serial_number))

            self.cam.MV_CC_SetIntValue('LineDebouncerTime', 2)
        elif use_software_trigger:
            ret = self.cam.MV_CC_SetEnumValueByString('TriggerMode', 'On')
            if ret != 0:
                raise Exception('set trigger mode on fail[0x%x], serial=%s' % (ret, self.serial_number))

            ret = self.cam.MV_CC_SetEnumValueByString('TriggerSource', 'Software')
            if ret != 0:
                raise Exception('set trigger source software fail[0x%x], serial=%s' % (ret, self.serial_number))
        else:
            ret = self.cam.MV_CC_SetEnumValue('TriggerMode', MV_TRIGGER_MODE_OFF)
            if ret != 0:
                raise Exception('set trigger mode off fail[0x%x], serial=%s' % (ret, self.serial_number))

        ret = self.cam.MV_CC_SetBayerCvtQuality(1)
        if ret != 0:
            print('[%s] warning: set bayer quality fail[0x%x]' % (self.serial_number, ret))

        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            raise Exception('start grabbing fail[0x%x], serial=%s' % (ret, self.serial_number))
        self.started_grab = True

        return self.readback_capture_params()

    def _try_set_enum_by_string(self, node_name, value_name):
        ret = self.cam.MV_CC_SetEnumValueByString(node_name, value_name)
        return ret == 0, ret

    def disable_auto_features(self):
        feature_candidates = {
            'ExposureAuto': ['Off'],
            'GainAuto': ['Off'],
            'BalanceWhiteAuto': ['Off'],
            'BalanceRatioAuto': ['Off'],
            'WhiteBalanceAuto': ['Off'],
            'FocusAuto': ['Off'],
        }

        for feature_name, values in feature_candidates.items():
            success = False
            last_ret = 0
            for value in values:
                ok, ret = self._try_set_enum_by_string(feature_name, value)
                last_ret = ret
                if ok:
                    success = True
                    break

            if success:
                print('[%s] %s=Off' % (self.serial_number, feature_name))
            else:
                print('[%s] %s unsupported or set failed[0x%x]' % (self.serial_number, feature_name, last_ret))

    def apply_manual_params(self, exposure_us=None, gain=None, frame_rate=None):
        if exposure_us is not None:
            ret = self.cam.MV_CC_SetFloatValue('ExposureTime', float(exposure_us))
            if ret != 0:
                raise Exception('set ExposureTime fail[0x%x], serial=%s' % (ret, self.serial_number))

        if gain is not None:
            ret = self.cam.MV_CC_SetFloatValue('Gain', float(gain))
            if ret != 0:
                raise Exception('set Gain fail[0x%x], serial=%s' % (ret, self.serial_number))

        if frame_rate is not None:
            ret = self.cam.MV_CC_SetFloatValue('AcquisitionFrameRate', float(frame_rate))
            if ret != 0:
                self.cam.MV_CC_SetBoolValue('AcquisitionFrameRateEnable', True)
                ret = self.cam.MV_CC_SetFloatValue('AcquisitionFrameRate', float(frame_rate))
            if ret != 0:
                raise Exception('set AcquisitionFrameRate fail[0x%x], serial=%s' % (ret, self.serial_number))

    def _get_float_value(self, node_name):
        st_float = MVCC_FLOATVALUE()
        memset(byref(st_float), 0, sizeof(MVCC_FLOATVALUE))
        ret = self.cam.MV_CC_GetFloatValue(node_name, st_float)
        if ret != 0:
            return None, ret
        return float(st_float.fCurValue), 0

    def readback_capture_params(self):
        result = {'serial': self.serial_number}
        for node_name, alias in [
            ('ExposureTime', 'exposure_us'),
            ('Gain', 'gain'),
            ('AcquisitionFrameRate', 'frame_rate'),
            ('ResultingFrameRate', 'resulting_frame_rate'),
        ]:
            value, ret = self._get_float_value(node_name)
            if ret == 0:
                result[alias] = value
                print('[%s] %s=%.6f' % (self.serial_number, node_name, value))
            else:
                result[alias] = None
                print('[%s] %s readback unsupported or failed[0x%x]' % (self.serial_number, node_name, ret))
        return result

    def grab_one_rgb(self, timeout_ms=1500):
        st_out_frame = MV_FRAME_OUT()
        memset(byref(st_out_frame), 0, sizeof(MV_FRAME_OUT))

        ret = self.cam.MV_CC_GetImageBuffer(st_out_frame, timeout_ms)
        if ret != 0 or st_out_frame.pBufAddr is None:
            raise Exception('get image fail[0x%x], serial=%s' % (ret, self.serial_number))

        try:
            width = st_out_frame.stFrameInfo.nWidth
            height = st_out_frame.stFrameInfo.nHeight
            rgb_size = width * height * 3

            st_convert_param = MV_CC_PIXEL_CONVERT_PARAM_EX()
            memset(byref(st_convert_param), 0, sizeof(MV_CC_PIXEL_CONVERT_PARAM_EX))
            st_convert_param.nWidth = width
            st_convert_param.nHeight = height
            st_convert_param.pSrcData = st_out_frame.pBufAddr
            st_convert_param.nSrcDataLen = st_out_frame.stFrameInfo.nFrameLen
            st_convert_param.enSrcPixelType = st_out_frame.stFrameInfo.enPixelType
            st_convert_param.enDstPixelType = PixelType_Gvsp_RGB8_Packed
            st_convert_param.pDstBuffer = (c_ubyte * rgb_size)()
            st_convert_param.nDstBufferSize = rgb_size

            ret = self.cam.MV_CC_ConvertPixelTypeEx(st_convert_param)
            if ret != 0:
                raise Exception('convert pixel fail[0x%x], serial=%s' % (ret, self.serial_number))

            rgb_bytes = string_at(st_convert_param.pDstBuffer, st_convert_param.nDstLen)
            image_rgb = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(height, width, 3)
            return image_rgb, int(st_out_frame.stFrameInfo.nFrameNum)
        finally:
            self.cam.MV_CC_FreeImageBuffer(st_out_frame)

    def software_trigger_once(self):
        ret = self.cam.MV_CC_SetCommandValue('TriggerSoftware')
        if ret != 0:
            raise Exception('software trigger fail[0x%x], serial=%s' % (ret, self.serial_number))

    def stop_and_close(self):
        if self.started_grab:
            self.cam.MV_CC_StopGrabbing()
            self.started_grab = False

        self.cam.MV_CC_CloseDevice()
        self.cam.MV_CC_DestroyHandle()


def compare_and_check_readbacks(readbacks, expected_value, key_name, tolerance, strict_check):
    valid = []
    for item in readbacks:
        v = item.get(key_name)
        if v is not None:
            valid.append(v)

    if not valid:
        print('warning: no readable values for %s on all cameras' % key_name)
        return

    min_v = min(valid)
    max_v = max(valid)
    print('summary %s range: min=%.6f, max=%.6f, diff=%.6f' % (key_name, min_v, max_v, max_v - min_v))

    if expected_value is not None and strict_check and abs(min_v - expected_value) > tolerance:
        raise Exception('%s check failed: min value %.6f differs from expected %.6f' % (key_name, min_v, expected_value))
    if expected_value is not None and strict_check and abs(max_v - expected_value) > tolerance:
        raise Exception('%s check failed: max value %.6f differs from expected %.6f' % (key_name, max_v, expected_value))
    if strict_check and (max_v - min_v) > tolerance:
        raise Exception('%s check failed: cross-camera mismatch diff %.6f > tolerance %.6f' % (key_name, max_v - min_v, tolerance))
