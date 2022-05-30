import numpy as np
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import grpc
import tensorflow as tf
from multiprocessing.dummy import Pool


def grpc_client_request(emb,
                        host='0.0.0.0',
                        port=8500,
                        emb_name='dense_1_input',
                        timeout=10):
    host = host.replace("http://", "").replace("https://", "")
    channel = grpc.insecure_channel("{}:{}".format(host, port))
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    # Create PredictRequest ProtoBuf from image data
    request = predict_pb2.PredictRequest()
    request.model_spec.name = "face_recognition"
    request.model_spec.signature_name = "serving_default"
    # image
    request.inputs[emb_name].CopyFrom(
        tf.contrib.util.make_tensor_proto(
            emb,
            dtype=np.float32,
        )
    )

    # Call the TFServing Predict API
    result = stub.Predict(request, timeout=timeout)
    return result


def api_recognition(emb):
    p = Pool(2)
    while True:
        result = p.map(grpc_client_request, [emb])
        break
    classes = result[0].outputs['dense_3'].float_val
    p.close()
    return classes

