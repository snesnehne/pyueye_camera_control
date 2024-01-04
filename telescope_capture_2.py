# Libraries
import ctypes

from pyueye import ueye
from pyueye_utils import (uEyeException, Rect, get_bits_per_pixel,
                                  ImageBuffer, check)

import numpy as np
import datetime as dt
import cv2 as cv
import time


def sample_definition(id_spec="BEN-B-Al-Al-002", load_mean=0, load_amp=1000, load_Hz=0):
    """Sample definition"""

    sd_dict = {"id_spec": id_spec, "load_mean": load_mean, "load_amp": load_amp, "load_Hz": load_Hz}

    print(f"SAMPLE DETAILS:"
          f"Specimen ID: {id_spec} \n"
          f"Mean load: {load_mean} N \n"
          f"Load Amplitude: {load_amp} N \n"
          f"Load Wave Frequency: {load_Hz} \n"
          f"-------------------------------------")

    return sd_dict


sd = sample_definition()  # modify sample definition if needed


class Telescope:
    def __init__(self):
        self.hCam = ueye.HIDS(0)  # 0: first available camera;  1-254: The camera with the specified camera ID
        self.sInfo = ueye.SENSORINFO()
        self.cInfo = ueye.CAMINFO()
        self.pcImageMemory = ueye.c_mem_p()
        self.MemID = ueye.int()
        self.rectAOI = ueye.IS_RECT()
        self.pitch = ueye.INT()
        self.nBitsPerPixel = ueye.INT(8)  # 24: bits per pixel for color mode; take 8 bits per pixel for monochrome
        self.channels = 1  # 3: channels for color mode(RGB); take 1 channel for monochrome
        self.m_nColorMode = ueye.INT()  # Y8/RGB16/RGB24/REG32
        self.bytes_per_pixel = int(self.nBitsPerPixel / 8)

        # Additional camera settings
        self.clock = None
        self.fps = None
        self.gain = None
        self.exposure = None

        # self.connect()
        # self.get_cam_info()
        # self.get_sensor_info()
        # self.set_color_mode()
        # self.set_aoi_original()

        # Starts the driver and establishes the connection to the camera
        self.open_c = ueye.is_InitCamera(self.hCam, None)
        if self.open_c != ueye.IS_SUCCESS:
            print("is_InitCamera ERROR")

        # Reads out the data hard-coded in the non-volatile camera memory and
        # writes it to the data structure that cInfo points to
        self.cam_info = ueye.is_GetCameraInfo(self.hCam, self.cInfo)
        if self.cam_info != ueye.IS_SUCCESS:
            print("is_GetCameraInfo ERROR")

        # You can query additional information about the sensor type used in the camera
        self.sen_info = ueye.is_GetSensorInfo(self.hCam, self.sInfo)
        if self.sen_info != ueye.IS_SUCCESS:
            print("is_GetSensorInfo ERROR")

        # Set the right color mode
        if int.from_bytes(self.sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_BAYER:
            # setup the color depth to the current windows setting
            ueye.is_GetColorDepth(self.hCam, self.nBitsPerPixel, self.m_nColorMode)
            self.bytes_per_pixel = int(self.nBitsPerPixel / 8)  # update by chu
            print("IS_COLORMODE_BAYER: ", )
            print("\tm_nColorMode: \t\t", self.m_nColorMode)
            print("\tnBitsPerPixel: \t\t", self.nBitsPerPixel)
            print("\tbytes_per_pixel: \t\t", self.bytes_per_pixel)
            print("-------------------------------------")

        elif int.from_bytes(self.sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_CBYCRY:
            # for color camera models use RGB32 mode
            self.m_nColorMode = ueye.IS_CM_BGRA8_PACKED
            self.nBitsPerPixel = ueye.INT(32)
            self.bytes_per_pixel = int(self.nBitsPerPixel / 8)
            print("IS_COLORMODE_CBYCRY: ", )
            print("\tm_nColorMode: \t\t", self.m_nColorMode)
            print("\tnBitsPerPixel: \t\t", self.nBitsPerPixel)
            print("\tbytes_per_pixel: \t\t", self.bytes_per_pixel)
            print("-------------------------------------")

        elif int.from_bytes(self.sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_MONOCHROME:
            # for color camera models use RGB32 mode
            self.m_nColorMode = ueye.IS_CM_MONO8  # added self to these variables
            self.nBitsPerPixel = ueye.INT(8)
            self.bytes_per_pixel = int(self.nBitsPerPixel / 8)
            print("IS_COLORMODE_MONOCHROME: ", )
            print("\tm_nColorMode: \t\t", self.m_nColorMode)
            print("\tnBitsPerPixel: \t\t", self.nBitsPerPixel)
            print("\tbytes_per_pixel: \t\t", self.bytes_per_pixel)
            print("-------------------------------------")

        else:
            # for monochrome camera models use Y8 mode
            self.m_nColorMode = ueye.IS_CM_MONO8
            self.nBitsPerPixel = ueye.INT(8)
            self.bytes_per_pixel = int(self.nBitsPerPixel / 8)
            print("else")

        # Can be used to set the size and position of an "area of interest"(AOI) within an image
        AOI = ueye.is_AOI(self.hCam, ueye.IS_AOI_IMAGE_GET_AOI, self.rectAOI, ueye.sizeof(self.rectAOI))
        if AOI != ueye.IS_SUCCESS:
            print("is_AOI ERROR")

        self.width = self.rectAOI.s32Width
        self.height = self.rectAOI.s32Height

        print("Camera model:\t\t", self.sInfo.strSensorName.decode('utf-8'))
        print("Camera serial no.:\t", self.cInfo.SerNo.decode('utf-8'))

        # Set other camera settings
        self.cam_settings()

    # <editor-fold desc="Settings">
    def set_aoi(self, x, y, width, height):
        """Set AOI (area of interest) position and size. Can allow for higher frame rate with less dropped frames"""
        rect_aoi = ueye.IS_RECT()
        rect_aoi.s32X = ueye.int(x)
        rect_aoi.s32Y = ueye.int(y)
        rect_aoi.s32Width = ueye.int(width)
        rect_aoi.s32Height = ueye.int(height)

        return ueye.is_AOI(self.hCam, ueye.IS_AOI_IMAGE_SET_AOI, rect_aoi, ueye.sizeof(rect_aoi))

    def set_aoi_original(self):
        """Can be used to set the size and position of an "area of interest"(AOI) within an image.
        This is original method in sample code"""
        AOI = ueye.is_AOI(self.hCam, ueye.IS_AOI_IMAGE_GET_AOI, self.rectAOI, ueye.sizeof(self.rectAOI))
        if AOI != ueye.IS_SUCCESS:
            print("is_AOI ERROR")

    def get_aoi(self):
        rect_aoi = ueye.IS_RECT()
        ueye.is_AOI(self.hCam, ueye.IS_AOI_IMAGE_GET_AOI, rect_aoi, ueye.sizeof(rect_aoi))

        return Rect(rect_aoi.s32X.value,
                    rect_aoi.s32Y.value,
                    rect_aoi.s32Width.value,
                    rect_aoi.s32Height.value)

    def set_pixel_clock(self, clock):
        """Sets pixel clock rate (MHz) of camera. A higher clock rate allows higher fps. Camera will generate more heat
        at higher clock rates and may require external cooling"""
        self.clock = clock
        pix_clock = ctypes.c_int(clock)
        pix_clock_byt = ctypes.c_byte(ctypes.sizeof(pix_clock))

        pixel_clock = ueye.is_PixelClock(self.hCam, ueye.IS_PIXELCLOCK_CMD_SET, pix_clock, pix_clock_byt)
        if pixel_clock != ueye.IS_SUCCESS:
            print("is_PixelClock ERROR")

    def get_clock(self):
        return self.clock

    def set_fps(self, fps):
        """Set fps (frames per second) to capture from the camera. Higher fps requires higher clock rate
        at a given resolution."""
        self.fps = fps
        fps = ctypes.c_double(fps)
        framerate = ueye.is_SetFrameRate(self.hCam, fps, ctypes.c_double())
        if framerate != ueye.IS_SUCCESS:
            print("is_Framerate ERROR")
        #self.get_fps = ueye.is_GetFramesPerSecond(hCam, ctypes.c_double())

    def get_fps(self):
        return self.fps

    def set_gain(self, gain):
        """Sets hardware gain (%) for camera. Using hardware gain is preferable to increasing brightness after
        an image is taken, but will still increase noise gain.
        Lighting should be optimized first before increasing gain"""
        self.gain = gain
        gainer = ueye.is_SetHardwareGain(self.hCam, ctypes.c_int(gain), ctypes.c_int(0), ctypes.c_int(0),
                                         ctypes.c_int(0))
        # gainer = ueye.is_SetHardwareGain(self.hCam, ctypes.c_double(gain), ctypes.c_double(0.0), ctypes.c_double(0.0),
        #                                  ctypes.c_double(0.0))

        # gainer = ueye.is_SetHardwareGain(self.hCam, ctypes.c_int(gain), ctypes.c_int(gain), ctypes.c_int(gain),
        #                                  ctypes.c_int(gain))

        if gainer != ueye.IS_SUCCESS:
            print("is_Gain ERROR")

    def get_gain(self):
        return self.gain

    def set_exposure(self, exposure):
        """Sets exposure time (ms) for camera. A shorter exposure time can reduce motion blur,
        but requires more light"""
        self.exposure = exposure
        exposure_tim = ueye.is_Exposure(self.hCam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, ctypes.c_double(exposure), ctypes.c_byte(8))
        if exposure_tim != ueye.IS_SUCCESS:
            print("is_Exposure ERROR")

    def get_exposure(self):
        return self.exposure

    def get_cam_info(self):
        # Reads out the data hard-coded in the non-volatile camera memory and writes it to the data structure that cInfo points to
        self.cam_info = ueye.is_GetCameraInfo(self.hCam, self.cInfo)
        if self.cam_info != ueye.IS_SUCCESS:
            print("is_GetCameraInfo ERROR")

    def get_sensor_info(self):
        # You can query additional information about the sensor type used in the camera
        self.sen_info = ueye.is_GetSensorInfo(self.hCam, self.sInfo)
        if self.sen_info != ueye.IS_SUCCESS:
            print("is_GetSensorInfo ERROR")

    def set_color_mode(self):
        # Set the right color mode
        if int.from_bytes(self.sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_BAYER:
            # setup the color depth to the current windows setting
            ueye.is_GetColorDepth(self.hCam, self.nBitsPerPixel, self.m_nColorMode)
            bytes_per_pixel = int(self.nBitsPerPixel / 8)  # updated by chu to nBitsPerPixel from BitsPerPixel
            print("IS_COLORMODE_BAYER: ", )
            print("\tm_nColorMode: \t\t", self.m_nColorMode)
            print("\tnBitsPerPixel: \t\t", self.nBitsPerPixel)
            print("\tbytes_per_pixel: \t\t", bytes_per_pixel)
            print()

        elif int.from_bytes(self.sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_CBYCRY:
            # for color camera models use RGB32 mode
            m_nColorMode = ueye.IS_CM_BGRA8_PACKED
            nBitsPerPixel = ueye.INT(32)
            bytes_per_pixel = int(nBitsPerPixel / 8)  # updated by chu to nBitsPerPixel from BitsPerPixel
            print("IS_COLORMODE_CBYCRY: ", )
            print("\tm_nColorMode: \t\t", m_nColorMode)
            print("\tnBitsPerPixel: \t\t", nBitsPerPixel)
            print("\tbytes_per_pixel: \t\t", bytes_per_pixel)
            print()

        elif int.from_bytes(self.sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_MONOCHROME:
            # for color camera models use RGB32 mode
            m_nColorMode = ueye.IS_CM_MONO8
            nBitsPerPixel = ueye.INT(8)
            bytes_per_pixel = int(nBitsPerPixel / 8)  # updated by chu to nBitsPerPixel from BitsPerPixel
            print("IS_COLORMODE_MONOCHROME: ", )
            print("\tm_nColorMode: \t\t", m_nColorMode)
            print("\tnBitsPerPixel: \t\t", nBitsPerPixel)
            print("\tbytes_per_pixel: \t\t", bytes_per_pixel)
            print()

        else:
            # for monochrome camera models use Y8 mode
            m_nColorMode = ueye.IS_CM_MONO8
            nBitsPerPixel = ueye.INT(8)
            bytes_per_pixel = int(nBitsPerPixel / 8)  # updated by chu to nBitsPerPixel from BitsPerPixel
            print("else")

    def def_settings(self):
        """Restore default camera settings"""
        default = ueye.is_ResetToDefault(self.hCam)
        if default != ueye.IS_SUCCESS:
            print("is_ResetToDefault ERROR")

    def cam_settings(self, clock=474, fps=20.0, gain=0, exposure=49, x=500, y=500, width=1000, height=1000):
        """Override some default camera settings and print out new settings"""
        # self.set_aoi(x, y, width, height)
        self.set_pixel_clock(clock)
        self.set_fps(fps)
        self.set_gain(gain)
        self.set_exposure(exposure)

        # Sets the position and size of the image by using an object of the IS_RECT type.
        #IS_RECT rectAOI;

        #rectAOI.s32X = 100;
        #rectAOI.s32Y = 100;
        #rectAOI.s32Width = 200;
        #rectAOI.s32Height = 100;

        #INT
        #nRet = is_AOI(hCam, IS_AOI_IMAGE_SET_AOI, (void *) & rectAOI, sizeof(rectAOI));
        # Can be used to set the size and position of an "area of interest"(AOI) within an image
        # rectAOI.s32Width
        #
        #
        # self.rectAOI = ueye.IS_RECT()
        #
        # AOI = ueye.is_AOI(self.hCam, ueye.IS_AOI_IMAGE_GET_AOI, self.rectAOI, ueye.sizeof(self.rectAOI))
        # if AOI != ueye.IS_SUCCESS:
        #     print("is_AOI ERROR")
        #
        self.width = self.rectAOI.s32Width
        self.height = self.rectAOI.s32Height

        print("Maximum image width:\t",self.width)
        print("Maximum image height:\t", self.height)

        print(f"Frame rate: {fps}")
        print(f"Pixel clock: {clock} MHz")
        print(f"Gain: {gain} %")
        print(f"Exposure time: {exposure} ms")

    # </editor-fold>

    def metadata(self, md_dict, md_fn):
        """Creates text file with metadata related to the test"""

        p_max = md_dict["load_mean"] + md_dict["load_amp"]  # max applied load (N)
        p_min = md_dict["load_mean"] - md_dict["load_amp"]  # min applied load (N)

        meta_data = f"Sample ID: {md_dict['sample']}\n" \
                    f"Sample test number: test{md_dict['instron_test_num']}\n" \
                    f"Start Time: {md_dict['start_tim']}\n" \
                    f"Starting Cycle: {md_dict['start_cyc']}\n" \
                    f"Cycle Frequency: {md_dict['loading_Hz']} Hz\n" \
                    f"Instron mean load: {md_dict['load_mean']} N\n" \
                    f"Instron load amplitude: {md_dict['load_amp']} N\n" \
                    f"Instron max load: {p_max} N\n" \
                    f"Instron min load: {p_min} N\n" \
                    f"Camera model number: {self.sInfo.strSensorName.decode('utf-8')}\n" \
                    f"Camera serial number: {self.cInfo.SerNo.decode('utf-8')}\n" \
                    f"Image Width: {self.width}, Image Height: {self.height}\n" \
                    f"fps: {self.get_fps()}\n" \
                    f"Pixel Clock rate: {self.get_clock()}\n" \
                    f"Exposure time: {self.get_exposure()}\n" \
                    f"Gain: {self.get_gain()}\n" \
                    #f"Gamma: {}\n" \
                    #f"Black-scale {}"
        # Write metadata text file
        with open(md_fn, "w") as writer:
            writer.write(meta_data)

        return meta_data

    def print_time(self, tot_time):
        print(f"Start time = {dt.datetime.now()}")
        print(f"Total capture time = {tot_time} m")
        print(f"Expected end time = {dt.datetime.now() + dt.timedelta(minutes=tot_time)}")

    def connect(self):
        # Starts the driver and establishes the connection to the camera
        self.open_c = ueye.is_InitCamera(self.hCam, None)
        if self.open_c != ueye.IS_SUCCESS:
            print("is_InitCamera ERROR")

    def reconnect(self):
        """Make a new connection to the camera.
        Must be used after the camera is closed at the end of a previous function"""
        self.connect()
        self.get_cam_info()
        self.get_sensor_info()
        self.set_color_mode()
        self.set_aoi_original()

        self.width = self.rectAOI.s32Width
        self.height = self.rectAOI.s32Height

        self.cam_settings()

    def close_cam(self):
        # Releases an image memory that was allocated using is_AllocImageMem() and removes it from the driver management
        ueye.is_FreeImageMem(self.hCam, self.pcImageMemory, self.MemID)
        # Disables the hCam camera handle and releases the data structures and memory areas taken up by the uEye camera
        ueye.is_ExitCamera(self.hCam)
        cv.destroyAllWindows()  # Destroys the OpenCv windows

    def user_inputs(self):
        """User inputs describing test"""
        start_cyc = int(input("Starting number of cycles: "))
        print("TEST WILL START AFTER ENTRY!!!")
        instron_test_num = int(input("Sample test number: "))

        input_dict = {"start_cyc": start_cyc, "instron_test_num": instron_test_num}

        return input_dict

    def allocate_memory(self):
        # Allocates an image memory for an image having its dimensions defined by width and height and its color depth defined by nBitsPerPixel
        nRet = ueye.is_AllocImageMem(self.hCam, self.width, self.height, self.nBitsPerPixel, self.pcImageMemory, self.MemID)
        if nRet != ueye.IS_SUCCESS:
            print("is_AllocImageMem ERROR")
        else:
            # Makes the specified image memory the active memory
            nRet = ueye.is_SetImageMem(self.hCam, self.pcImageMemory, self.MemID)
            if nRet != ueye.IS_SUCCESS:
                print("is_SetImageMem ERROR")
            else:
                # Set the desired color mode
                nRet = ueye.is_SetColorMode(self.hCam, self.m_nColorMode)

    def enable_que(self):
        # Enables the queue mode for existing image memory sequences
        nRet = ueye.is_InquireImageMem(self.hCam, self.pcImageMemory, self.MemID, self.width, self.height,
                                       self.nBitsPerPixel, self.pitch)
        if nRet != ueye.IS_SUCCESS:
            print("is_InquireImageMem ERROR")

    def enable_live_mode(self):
        # Activates the camera's live video mode (free run mode)
        nRet = ueye.is_CaptureVideo(self.hCam, ueye.IS_DONT_WAIT)
        if nRet != ueye.IS_SUCCESS:
            print("is_CaptureVideo ERROR")

    def vid_seq(self, tot_time=60, vid_len=4, vid_int=1, load_mean=1116, load_amp=913):
        """
        Records a sequence of videos with a given total time (minutes), video length (s), and video interval (minutes).
        Metadata about the settings used is also stored
        """
        u_input = self.user_inputs()

        start_tim = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        start_time_2 = time.time()

        md_fn = f"{sd['id_spec']}_t{u_input['instron_test_num']:02d}_c{u_input['start_cyc']:07d}_{start_tim}.txt"
        md_dict = {"sample": sd['id_spec'], "instron_test_num": u_input['instron_test_num'], "start_tim": start_tim,
                   "start_cyc": u_input['start_cyc'], "loading_Hz": sd['load_Hz'],
                   "load_mean": sd['load_mean'], "load_amp": sd['load_amp']}

        self.metadata(md_dict, md_fn)
        self.allocate_memory()
        self.enable_que()
        self.enable_live_mode()

        fourcc = cv.VideoWriter_fourcc(*'XVID')  # Define the codec

        vid_num = 0
        print("Press q to leave the program early")

        # Continue recording videos for set total time
        while time.time() < start_time_2 + (tot_time*60):
            # Start new video if time passes set interval
            if not time.time() < start_time_2 + (vid_int*60*vid_num):
                vid_start = time.time()
                vid_start_dt = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                elapsed_time = time.time() - vid_start

                fname = f"{sd['id_spec']}_t{u_input['instron_test_num']:02d}_c{u_input['start_cyc']:07d}" \
                        f"_v{vid_num:03d}_{vid_start_dt}.avi"
                out = cv.VideoWriter(fname, fourcc, self.fps, (self.width.value, self.height.value), 0)

                # Record video while elapsed time is less than specified video length
                while elapsed_time < vid_len:
                    # extract the data of our image memory
                    array = ueye.get_data(self.pcImageMemory, self.width, self.height, self.nBitsPerPixel, self.pitch, copy=False)
                    # ...reshape it in an numpy array...
                    frame = np.reshape(array, (self.height.value, self.width.value, self.bytes_per_pixel))
                    # frame_sm = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)
                    out.write(frame)
                    # cv.imshow("SimpleLive_Python_uEye_OpenCV", frame_sm)

                    elapsed_time = time.time() - vid_start

                out.release()
                # cv.destroyAllWindows()

                vid_num += 1
            # Press q if you want to end the loop
            if cv.waitKey(1) & 0xFF == ord('q'):
                self.close_cam()
                break
        self.close_cam()

    def vid_seq_display(self, tot_time=60, vid_len=4, vid_int=1):
        """
        Records a sequence of videos with a given total time (minutes), video length (s), and video interval (minutes).
        Live view of camera is displayed when not recording video to memory.
        Metadata about the settings used is also stored.
        """
        u_input = self.user_inputs()

        start_tim = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        start_time_2 = time.time()

        md_fn = f"{sd['id_spec']}_t{u_input['instron_test_num']:02d}_c{u_input['start_cyc']:07d}_{start_tim}.txt"
        md_dict = {"sample": sd['id_spec'], "instron_test_num": u_input['instron_test_num'], "start_tim": start_tim,
                   "start_cyc": u_input['start_cyc'], "loading_Hz": sd['load_Hz'],
                   "load_mean": sd['load_mean'], "load_amp": sd['load_amp']}
        self.metadata(md_dict, md_fn)
        self.allocate_memory()
        self.enable_que()
        self.enable_live_mode()

        fourcc = cv.VideoWriter_fourcc(*'XVID')  # Define the codec

        vid_num = 0
        start_cyc = u_input['start_cyc']
        print("Press q to leave the program early")
        self.print_time(tot_time=tot_time)

        # Continue recording videos for set total time
        while time.time() < start_time_2 + (tot_time*60):
            array = ueye.get_data(self.pcImageMemory, self.width, self.height, self.nBitsPerPixel, self.pitch,
                                  copy=False)  # extract the data of our image memory
            frame = np.reshape(array, (self.height.value, self.width.value, self.bytes_per_pixel))  # reshape as array
            frame_sm = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)  # resize array
            # cv.imshow("Telescope View, not recording", frame_sm)  # display image on screen
            cv.imshow("Telescope View", frame_sm)  # display image on screen

            # Start new video recording if time passes set interval
            if not time.time() < start_time_2 + (vid_int*60*vid_num):
                # cv.destroyWindow("Telescope View, not recording")
                vid_start = time.time()
                vid_start_dt = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                elapsed_time = time.time() - vid_start
                cur_cycle = start_cyc + (10*60*vid_int)

                fname = f"{sd['id_spec']}_t{u_input['instron_test_num']:02d}_c{cur_cycle}" \
                        f"_v{vid_num:03d}_{vid_start_dt}.avi"
                out = cv.VideoWriter(fname, fourcc, self.fps, (self.width.value, self.height.value), 0)

                # Record video while elapsed time is less than specified video length
                while elapsed_time < vid_len:
                    out.write(frame)
                    elapsed_time = time.time() - vid_start
                #     frame_sm2 = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)  # resize array
                #     cv.imshow("Telescope View, recording", frame_sm2)  # display image on screen
                #
                # cv.destroyWindow("Telescope View, recording")
                out.release()

                vid_num += 1


            # Press q if you want to end the loop
            if cv.waitKey(1) & 0xFF == ord('q'):
                self.close_cam()
                print(f'Ended early at: {dt.datetime.now()}')

                break
        print(f'Finished: {dt.datetime.now()}')
        # Write metadata text file
        with open(md_fn, "a") as writer:
            writer.write(f'Ended at: {dt.datetime.now()}\n'
                         f'Elapsed time (s): {time.time()-start_time_2}')
        self.close_cam()

    def vid_single(self, vid_len=10):
        """Records a video of length vid_len (minutes). Metadata about the settings used is also stored"""
        u_input = self.user_inputs()
        self.allocate_memory()
        self.enable_que()
        self.enable_live_mode()

        fourcc = cv.VideoWriter_fourcc(*'XVID')  # Define the codec

        start_tim = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        start_time_2 = time.time()

        md_fn = f"{sd['id_spec']}_t{u_input['instron_test_num']:02d}_c{u_input['start_cyc']:07d}_{start_tim}.txt"
        md_dict = {"sample": sd['id_spec'], "instron_test_num": u_input['instron_test_num'], "start_tim": start_tim,
                   "start_cyc": u_input['start_cyc'], "loading_Hz": sd['load_Hz'],
                   "load_mean": sd['load_mean'], "load_amp": sd['load_amp']}
        self.metadata(md_dict, md_fn)

        vid_num = 0
        print("Press q to leave the program early")
        self.print_time(tot_time=vid_len)

        fname = f"{sd['id_spec']}_t{u_input['instron_test_num']:02d}_c{u_input['start_cyc']:07d}" \
                f"_v{vid_num:03d}_{start_tim}.avi"
        out = cv.VideoWriter(fname, fourcc, self.fps, (self.width.value, self.height.value), 0)

        vid_start = time.time()
        elapsed_time = time.time() - vid_start

        # Record video while elapsed time is less than specified video length
        while elapsed_time < vid_len*60:
            # extract the data of our image memory
            array = ueye.get_data(self.pcImageMemory, self.width, self.height, self.nBitsPerPixel, self.pitch,
                                  copy=False)
            # ...reshape it in an numpy array...
            frame = np.reshape(array, (self.height.value, self.width.value, self.bytes_per_pixel))
            # frame_sm = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)

            out.write(frame)
            frame_sm = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)  # resize array
            cv.imshow("Telescope View, recording", frame_sm)  # display image on screen
            # cv.imshow("SimpleLive_Python_uEye_OpenCV", frame_sm)

            elapsed_time = time.time() - vid_start

            # Press q if you want to end the loop
            if cv.waitKey(1) & 0xFF == ord('q'):
                self.close_cam()
                break

        out.release()
        with open(md_fn, "a") as writer:
            writer.write(f'Ended at: {dt.datetime.now()}\n'
                         f'Elapsed time (s): {time.time()-start_time_2}')
        # cv.destroyAllWindows()
        self.close_cam()

    def framebyframe(self, tot_time=60, tim2frame=1, load_mean=900, load_amp=700): ####UNDER CONTSRUCTION!!
        """
        Records a sequence of videos with a given total time (minutes), video length (s), and video interval (minutes).
        Metadata about the settings used is also stored
        """
        # User inputs describing test
        sample = input("Sample ID: (ex. BEN-B-Al-Al-002)")
        start_cyc = input("Starting number of cycles: ")
        loading_Hz = 10  # (1/s)
        print("Test will start after entry")
        instron_test_num = input("Sample test number: ")

        # Allocates an image memory for an image having its dimensions defined by width and height and its color depth defined by nBitsPerPixel
        nRet = ueye.is_AllocImageMem(self.hCam, self.width, self.height, self.nBitsPerPixel, self.pcImageMemory, self.MemID)
        if nRet != ueye.IS_SUCCESS:
            print("is_AllocImageMem ERROR")
        else:
            # Makes the specified image memory the active memory
            nRet = ueye.is_SetImageMem(self.hCam, self.pcImageMemory, self.MemID)
            if nRet != ueye.IS_SUCCESS:
                print("is_SetImageMem ERROR")
            else:
                # Set the desired color mode
                nRet = ueye.is_SetColorMode(self.hCam, self.m_nColorMode)

        # Activates the camera's live video mode (free run mode)
        nRet = ueye.is_CaptureVideo(self.hCam, ueye.IS_DONT_WAIT)
        if nRet != ueye.IS_SUCCESS:
            print("is_CaptureVideo ERROR")

        # Enables the queue mode for existing image memory sequences
        nRet = ueye.is_InquireImageMem(self.hCam, self.pcImageMemory, self.MemID, self.width, self.height,
                                       self.nBitsPerPixel, self.pitch)
        if nRet != ueye.IS_SUCCESS:
            print("is_InquireImageMem ERROR")

        img_num = 0
        print("Press q to leave the program early")

        start_tim = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # start_time_2 = time.time()

        meta_data_fn = f"{sample}_c{start_cyc}_{start_tim}.txt"
        meta_data = f"Sample ID: {sample}" \
                    f"Sample test number: test{instron_test_num}\n" \
                    f"Start Time: {start_tim}\n" \
                    f"Starting Cycle: {start_cyc}\n" \
                    f"Cycle Frequency: {loading_Hz} Hz\n" \
                    f"Instron mean load: {load_mean} N\n" \
                    f"Instron load amplitude: {load_amp} N\n" \
                    f"Camera info: {self.cam_info}\n" \
                    f"Sensor info: {self.sen_info}\n" \
                    f"Image Width: {self.width}, Image Height: {self.height}\n" \
                    f"fps: {self.get_fps()}\n" \
                    f"Pixel Clock rate: {self.get_clock()}\n" \
                    f"Exposure time: {self.get_exposure()}\n" \
                    f"Gain: {self.get_gain()}\n" \
                    #f"Gamma: {}\n" \
                    #f"Black-scale {}"

        # Write metadata text file
        with open(meta_data_fn, "w") as writer:
            writer.write(meta_data)

        start_time_2 = time.time()

        # Continue recording videos for set total time
        while time.time() < start_time_2 + (tot_time*60):
            vid_start_tim = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            fname = f"{sample}_t{instron_test_num}_c{start_cyc}_i{img_num}_{vid_start_tim}.jpg"

            # extract the data of our image memory
            array = ueye.get_data(self.pcImageMemory, self.width, self.height, self.nBitsPerPixel, self.pitch, copy=False)
            # ...reshape it in an numpy array...
            frame = np.reshape(array, (self.height.value, self.width.value, self.bytes_per_pixel))
            # frame_sm = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)

            cv.imwrite(fname, frame)

            img_num += 1

            time.sleep(tim2frame)
            # Press q if you want to end the loop
            if cv.waitKey(1) & 0xFF == ord('q'):
                self.close_cam()
                break
        self.close_cam()

    def live_video(self):
        """Creates a live window of the images being returned from the camera. Does not save"""
        self.allocate_memory()
        self.allocate_memory()
        self.enable_que()
        self.enable_live_mode()

        print("Press q to leave the program early")

        # Continuous image display
        # while (nRet == ueye.IS_SUCCESS):
        while True:
            # extract the data of our image memory
            array = ueye.get_data(self.pcImageMemory, self.width, self.height, self.nBitsPerPixel, self.pitch, copy=False)
            # reshape data in an numpy array
            frame = np.reshape(array, (self.height.value, self.width.value, self.bytes_per_pixel))
            frame = cv.resize(frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)  # resize array
            # display image on computer screen
            cv.imshow(f"Telescope View", frame)

            # Press q if you want to end the loop
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        self.close_cam()

    def save_till_quit(self):
        fname = input("Enter filename: ")

        # Allocates an image memory for an image having its dimensions defined by width and height and
        # its color depth defined by nBitsPerPixel
        nRet = ueye.is_AllocImageMem(self.hCam, self.width, self.height, self.nBitsPerPixel, self.pcImageMemory, self.MemID)
        if nRet != ueye.IS_SUCCESS:
            print("is_AllocImageMem ERROR")
        else:
            # Makes the specified image memory the active memory
            nRet = ueye.is_SetImageMem(self.hCam, self.pcImageMemory, self.MemID)
            if nRet != ueye.IS_SUCCESS:
                print("is_SetImageMem ERROR")
            else:
                # Set the desired color mode
                nRet = ueye.is_SetColorMode(self.hCam, self.m_nColorMode)

        # Activates the camera's live video mode (free run mode)
        nRet = ueye.is_CaptureVideo(self.hCam, ueye.IS_DONT_WAIT)
        if nRet != ueye.IS_SUCCESS:
            print("is_CaptureVideo ERROR")

        # Enables the queue mode for existing image memory sequences
        nRet = ueye.is_InquireImageMem(self.hCam, self.pcImageMemory, self.MemID, self.width, self.height,
                                       self.nBitsPerPixel, self.pitch)
        if nRet != ueye.IS_SUCCESS:
            print("is_InquireImageMem ERROR")
        else:
            print("Press q to leave the program")

        # ---------------------------------------------------------------------------------------------------------------------------------------
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        out = cv.VideoWriter(fname, fourcc, self.fps, (self.width.value, self.height.value), 0)
        # out = cv.VideoWriter('output_gray_10.avi', fourcc, 60.0, (2456, 2054), 0)
        # Continuous image display
        while (nRet == ueye.IS_SUCCESS):
            # ...extract the data of our image memory
            array = ueye.get_data(self.pcImageMemory, self.width, self.height, self.nBitsPerPixel, self.pitch, copy=False)

            # ...reshape it in an numpy array...
            frame = np.reshape(array, (self.height.value, self.width.value, self.bytes_per_pixel))

            # ...resize the image by a half
            frame_sm = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)

            cv.imshow("SimpleLive_Python_uEye_OpenCV", frame_sm)
            out.write(frame)

            # Press q if you want to end the loop
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        # Releases an image memory that was allocated using is_AllocImageMem() and removes it from the driver management
        ueye.is_FreeImageMem(self.hCam, self.pcImageMemory, self.MemID)

        # Disables the hCam camera handle and releases the data structures and memory areas taken up by the uEye camera
        ueye.is_ExitCamera(self.hCam)

        # Destroys the OpenCv windows
        cv.destroyAllWindows()
        out.release()

t = Telescope()

# cam = Camera()
# t.cam_settings()