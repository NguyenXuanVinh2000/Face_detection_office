import cv2
import numpy as np
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import grpc
import tensorflow as tf
from multiprocessing.dummy import Pool


def grpc_client_request(img,
                        host='0.0.0.0',
                        port=8500,
                        img_name='input',
                        min_size='min_size',
                        thresholds='thresholds',
                        factor='factor',
                        timeout=10):
    host = host.replace("http://", "").replace("https://", "")
    channel = grpc.insecure_channel("{}:{}".format(host, port))
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    # Create PredictRequest ProtoBuf from image data
    request = predict_pb2.PredictRequest()
    request.model_spec.name = "face_detection"
    request.model_spec.signature_name = "serving_default"
    # image
    request.inputs[img_name].CopyFrom(
        tf.contrib.util.make_tensor_proto(
            img,
            dtype=np.float32,
            shape=img.shape,
        )
    )

    request.inputs[min_size].CopyFrom(
        tf.contrib.util.make_tensor_proto(
            40,
            dtype=np.float32,
        )
    )

    request.inputs[thresholds].CopyFrom(
        tf.contrib.util.make_tensor_proto(
            [0.6, 0.7, 0.7],
            dtype=np.float32,
        )
    )

    request.inputs[factor].CopyFrom(
        tf.contrib.util.make_tensor_proto(
            0.6,
            dtype=np.float32
        )
    )

    # Call the TFServing Predict API
    result = stub.Predict(request, timeout=timeout)

    return result


def api_detector(img):
    p = Pool(2)
    while True:
        result = p.map(grpc_client_request, [img])
        break
    box = result[0].outputs['box'].float_val
    landmarks = result[0].outputs['landmarks'].float_val
    confidence = result[0].outputs['prob'].float_val


    p.close()
    return box, landmarks, confidence
