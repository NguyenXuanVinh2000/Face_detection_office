from tensorflow import keras

model = keras.models.load_model('/home/vinh/Face-Recognition-with-InsightFace/src/outputs/my_model.h5')

keras.experimental.export_saved_model(model, '/home/vinh/Face-Recognition-with-InsightFace/Vinh_scr/output')
