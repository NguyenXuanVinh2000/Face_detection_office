import videostream
from threading import Thread
import numpy as np
import cv2
#from server_detection import api_detector


def do_stuff(q):
  while True:
    global index
    img, ind = q.get()
    if index % 3 == 0:
        img = cv2.resize(img, (500, 600))
    if ind > index:
        #cv2.imshow("camera", img)
        #cv2.waitKey(1)
        index = ind
        cv2.imshow("Camera", img)
        cv2.waitKey(1)
        print(ind)
        q.task_done()

    else:
        q.task_done()
        return


num_threads = 10
index = 0
scr = "rtsp://admin:ervai!@#123@172.16.14.27/"
scr2 = 0
cap = videostream.FileVideoStream(scr).start()

# cap1 = videostream.FileVideoStream(scr2).start()
#while cap.more():
for i in range(num_threads):
        worker = Thread(target=do_stuff, args=(cap.Q, ))
        worker.setDaemon(True)
        worker.start()

#while cap.more():
# frame2, index2 = cap1.Q.get()
# frame = np.concatenate((cv2.resize(frame, (700, 500)), cv2.resize(frame2, (700, 500))), axis=1)
# do_stuff(cap.Q)


cap.Q.join()
