from Threading import VideoThreading


if __name__ == '__main__':

    scr_camera1 = "rtsp://admin:ervai!@#123@172.16.14.27:554/Streaming/Channels/101"
    scr_camera2 = "rtsp://admin:ervai!@#123@172.16.14.21:554"
    camera1 = VideoThreading(scr_camera1)
    camera2 = VideoThreading(scr_camera2)
    while True:
        try:
            camera1.show_frame("camera1")
            camera2.show_frame("camera2")
        except AttributeError:
            pass