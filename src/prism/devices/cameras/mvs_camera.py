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

    def open_and_prepare(self, exposure_us=None, gain=None, frame_rate=None, trigger_source='Line0',
                         gpio_output_line=None):
        """
        Initialize and prepare camera for hardware-triggered capture.

        Args:
            exposure_us: Exposure time in microseconds
            gain: Gain value
            frame_rate: Acquisition frame rate (fps)
            trigger_source: 'Software' (master) or 'Line0' (slave hardware trigger)
            gpio_output_line: GPIO line number for strobe output (master only, typically 1)
        """
        ret = self.cam.MV_CC_CreateHandle(self.device_info)
        if ret != 0:
            raise Exception('create handle fail[0x%x], serial=%s' % (ret, self.serial_number))

        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            raise Exception('open device fail[0x%x], serial=%s' % (ret, self.serial_number))

        self.disable_auto_features()
        self.apply_manual_params(exposure_us=exposure_us, gain=gain, frame_rate=frame_rate)

        # Configure trigger mode (Software for master, Line0 for slaves)
        self.cam.MV_CC_SetEnumValueByString('TriggerSelector', 'FrameStart')  # best-effort

        ret = self.cam.MV_CC_SetEnumValueByString('TriggerMode', 'On')
        if ret != 0:
            raise Exception('set trigger mode on fail[0x%x], serial=%s' % (ret, self.serial_number))

        ret = self.cam.MV_CC_SetEnumValueByString('TriggerSource', trigger_source)
        if ret != 0:
            raise Exception('set trigger source %s fail[0x%x], serial=%s' % (trigger_source, ret, self.serial_number))

        # For hardware trigger inputs (slave cameras), trigger on falling edge of Line0
        if trigger_source not in ('Software',):
            self.cam.MV_CC_SetEnumValueByString('TriggerActivation', 'FallingEdge')
            self.cam.MV_CC_SetIntValue('LineDebouncerTime', 2)
        
        # Configure GPIO output for triggering other cameras if specified
        if gpio_output_line is not None:
            self._configure_gpio_output(gpio_output_line)

        ret = self.cam.MV_CC_SetBayerCvtQuality(1)
        if ret != 0:
            print('[%s] warning: set bayer quality fail[0x%x]' % (self.serial_number, ret))

        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            raise Exception('start grabbing fail[0x%x], serial=%s' % (ret, self.serial_number))
        self.started_grab = True

        return self.readback_capture_params()

    def _configure_gpio_output(self, line_number):
        """
        Configure GPIO line as Strobe output on the master camera.

        Strobe source = ExposureActive: signal goes HIGH the moment the master
        camera starts its own exposure, so slave cameras on Line0 all start
        exposure at the exact same instant.

        GenICam nodes used:
            LineSelector  (enum)  – select the physical line
            LineMode      (enum)  – "Strobe"
            LineSource    (enum)  – "ExposureActive"
            StrobeEnable  (bool)  – enable strobe output

        Args:
            line_number: GPIO line number (1 for master Line1 output)
        """
        line_name = 'Line%d' % line_number

        # 1. Select the line
        ret = self.cam.MV_CC_SetEnumValueByString('LineSelector', line_name)
        if ret != 0:
            print('[%s] warning: LineSelector=%s failed[0x%x]' % (self.serial_number, line_name, ret))
            return

        # 2. Set LineMode to Strobe (only supported output mode on Line1 for MV-CS004-10UC)
        ret = self.cam.MV_CC_SetEnumValueByString('LineMode', 'Strobe')
        if ret != 0:
            print('[%s] warning: LineMode=Strobe failed[0x%x]' % (self.serial_number, ret))
            return

        # 3. Set strobe source to FrameTriggerWait:
        #    HIGH while master waits for software trigger, goes LOW (falling edge) the instant
        #    the master receives the trigger and begins exposure.
        #    Slave cameras trigger on this falling edge -> all cameras expose simultaneously.
        ret = self.cam.MV_CC_SetEnumValueByString('LineSource', 'FrameTriggerWait')
        if ret != 0:
            print('[%s] warning: LineSource=FrameTriggerWait failed[0x%x]' % (self.serial_number, ret))
            return

        # 4. Enable strobe output
        ret = self.cam.MV_CC_SetBoolValue('StrobeEnable', True)
        if ret != 0:
            print('[%s] warning: StrobeEnable=True failed[0x%x]' % (self.serial_number, ret))
            return

        print('[%s] GPIO %s configured as Strobe output (source=FrameTriggerWait, slaves use FallingEdge)' % (self.serial_number, line_name))


    def software_trigger_once(self):
        """Send one software trigger pulse to the master camera."""
        self.cam.MV_CC_SetCommandValue('TriggerSoftware')

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

    def grab_one_bgr(self, timeout_ms=1500):
        st_out_frame = MV_FRAME_OUT()
        memset(byref(st_out_frame), 0, sizeof(MV_FRAME_OUT))

        ret = self.cam.MV_CC_GetImageBuffer(st_out_frame, timeout_ms)
        if ret != 0 or st_out_frame.pBufAddr is None:
            raise Exception('get image fail[0x%x], serial=%s' % (ret, self.serial_number))

        try:
            width = st_out_frame.stFrameInfo.nWidth
            height = st_out_frame.stFrameInfo.nHeight
            bgr_size = width * height * 3

            st_convert_param = MV_CC_PIXEL_CONVERT_PARAM_EX()
            memset(byref(st_convert_param), 0, sizeof(MV_CC_PIXEL_CONVERT_PARAM_EX))
            st_convert_param.nWidth = width
            st_convert_param.nHeight = height
            st_convert_param.pSrcData = st_out_frame.pBufAddr
            st_convert_param.nSrcDataLen = st_out_frame.stFrameInfo.nFrameLen
            st_convert_param.enSrcPixelType = st_out_frame.stFrameInfo.enPixelType
            st_convert_param.enDstPixelType = PixelType_Gvsp_BGR8_Packed
            st_convert_param.pDstBuffer = (c_ubyte * bgr_size)()
            st_convert_param.nDstBufferSize = bgr_size

            ret = self.cam.MV_CC_ConvertPixelTypeEx(st_convert_param)
            if ret != 0:
                raise Exception('convert pixel fail[0x%x], serial=%s' % (ret, self.serial_number))

            bgr_bytes = string_at(st_convert_param.pDstBuffer, st_convert_param.nDstLen)
            image_bgr = np.frombuffer(bgr_bytes, dtype=np.uint8).reshape(height, width, 3)
            return image_bgr, int(st_out_frame.stFrameInfo.nFrameNum)
        finally:
            self.cam.MV_CC_FreeImageBuffer(st_out_frame)

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
