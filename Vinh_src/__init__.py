from queue_thread import *
# vid = cv2.VideoCapture("rtsp://admin:ervai!@#123@172.16.14.27/")
vid = cv2.VideoCapture(0)
#vid2 = cv2.VideoCapture("rtsp://admin:ervai!@#123@172.16.14.21/")
vid2 = cv2.VideoCapture("/home/vinh/Face-Recognition-with-InsightFace/Vinh_src/videotest3.mp4")
ret, frame = vid.read()
windows_camera = Recognition(frame, frame).start()
while True:
    ret, frame = vid.read()
    ret2, frame2 = vid2.read()

    if (not ret) or (not ret2) or windows_camera.stopped:
        windows_camera.stop()
        break

    windows_camera.frame = frame
    windows_camera.frame2 = frame2

vid.release()
# vid2.release()
cv2.destroyAllWindows()
