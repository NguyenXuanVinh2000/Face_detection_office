# Face_detection_office
  The project building the face recognition system: face detection MTCNN and face recognition InsightFace. And use microservice tensorflow-serving for the system.
# Run project
  - Clone project
  - Install all package in project.
  - Install docker, tensorflow serving
## Command the folder: Vinh_scr/docker. 
  - Build image by: docker build -t face_recognition_gpu:1.15.0-gpu
  - Run server by: docker-compose up
## In the Folder Vinh_scr.
  - Edit path to the camera in the client_threading.py
  - pyhton3 client_threading.py
