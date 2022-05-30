import sys
import threading

sys.path.append('../insightface/deploy')
sys.path.append('../insightface/src/common')
import server_recognition
import server_detection
import face_preprocess
import numpy as np
import face_model
import argparse
import pickle
import time
import dlib
import cv2
import multiprocessing

ap = argparse.ArgumentParser()

ap.add_argument("--le", default="outputs/le.pickle",
                help="Path to label encoder")
ap.add_argument("--embeddings", default="outputs/embeddings.pickle",
                help='Path to embeddings')

ap.add_argument('--image-size', default='112,112', help='')
ap.add_argument('--model', default='../insightface/models/model-y1-test2/model,0', help='path to load model.')
ap.add_argument('--ga-model', default='', help='path to load model.')
ap.add_argument('--gpu', default=0, type=int, help='gpu id')
ap.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
ap.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
ap.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')

args = ap.parse_args()

# Load embeddings and labels
data = pickle.loads(open(args.embeddings, "rb").read())
le = pickle.loads(open(args.le, "rb").read())

embeddings = np.array(data['embeddings'])
labels = le.fit_transform(data['names'])

# Initialize faces embedding model
embedding_model = face_model.FaceModel(args)


# Load the classifier model


def findCosineDistance(vector1, vector2):
    """
    Calculate cosine distance between two vector
    """
    vec1 = vector1.flatten()
    vec2 = vector2.flatten()

    a = np.dot(vec1.T, vec2)
    b = np.dot(vec1.T, vec1)
    c = np.dot(vec2.T, vec2)
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


def CosineSimilarity(test_vec, source_vecs):
    """
    Verify the similarity of one vector to group vectors of one class
    """
    cos_dist = 0
    for source_vec in source_vecs:
        cos_dist += findCosineDistance(test_vec, source_vec)
    return cos_dist / len(source_vecs)


# Start streaming and recording
def camera1():
    # Initialize some useful arguments
    print("start1")
    cosine_threshold = 0.8
    proba_threshold = 0.85
    comparing_num = 5
    trackers = []
    texts = []
    frames = 0
    # camera
    scr = "/home/vinh/Face-Recognition-with-InsightFace/Vinh_src/videotest2.mp4"
    cap = cv2.VideoCapture(scr)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    save_width = 600
    save_height = int(600 / frame_width * frame_height)

    while True:
        ret, frame = cap.read()
        frames += 1
        if frames % 3 == 1:
            start_time = time.time()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (save_width, save_height))
        if frames % 3 == 0:
            trackers = []
            texts = []

            box, landmark, confidence = server_detection.server.grpc_client_request(frame)

            if len(box) > 0:
                for i in range(0, len(confidence)):
                    # bounding box
                    top = int(box[i * 4])
                    left = int(box[i * 4 + 1])
                    bottom = int(box[i * 4 + 2])
                    right = int(box[i * 4 + 3])
                    bbox = np.array([left, top, right, bottom])
                    # landmarks
                    left_eye1 = int(landmark[i * 10])
                    right_eye1 = int(landmark[i * 10 + 1])
                    nose1 = int(landmark[i * 10 + 2])
                    mouth_left1 = int(landmark[i * 10 + 3])
                    mouth_right1 = int(landmark[i * 10 + 4])
                    left_eye0 = int(landmark[i * 10 + 5])
                    right_eye0 = int(landmark[i * 10 + 6])
                    nose0 = int(landmark[i * 10 + 7])
                    mouth_left0 = int(landmark[i * 10 + 8])
                    mouth_right0 = int(landmark[i * 10 + 9])
                    landmarks = np.array(
                        [left_eye0, right_eye0, nose0, mouth_left0, mouth_right0, left_eye1, right_eye1, nose1,
                         mouth_left1, mouth_right1])

                    landmarks = landmarks.reshape((2, 5)).T
                    nimg = face_preprocess.preprocess(frame, bbox, landmarks, image_size='112,112')
                    nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
                    nimg = np.transpose(nimg, (2, 0, 1))
                    embedding = embedding_model.get_feature(nimg).reshape(1, -1)

                    text = "Unknown"

                    # Predict class
                    preds = server_recognition.server.grpc_client_request(embedding)

                    # Get the highest accuracy embedded vector
                    j = np.argmax(preds)
                    j = int(j)
                    proba = preds[j]
                    # Compare this vector to source class vectors to verify it is actual belong to this class
                    match_class_idx = (labels == j)
                    match_class_idx = np.where(match_class_idx)[0]
                    selected_idx = np.random.choice(match_class_idx, comparing_num)
                    compare_embeddings = embeddings[selected_idx]
                    # Calculate cosine similarity
                    cos_similarity = CosineSimilarity(embedding, compare_embeddings)
                    if cos_similarity < cosine_threshold and proba > proba_threshold:
                        name = le.classes_[j]
                        text = "{}".format(name)
                    # Start tracking
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(bbox[0], bbox[1], bbox[2], bbox[3])
                    tracker.start_track(rgb, rect)
                    trackers.append(tracker)
                    texts.append(text)

                    y = bbox[1] - 10 if bbox[1] - 10 > 10 else bbox[1] + 10
                    cv2.putText(frame, text, (bbox[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                    FPS = 3 / (time.time() - start_time)
                    fps = str(round(FPS))
                    cv2.putText(frame, fps, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            for tracker, text in zip(trackers, texts):
                pos = tracker.get_position()

                # unpack the position object
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
                cv2.putText(frame, text, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        cv2.imshow("Camera1", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


camera1()
