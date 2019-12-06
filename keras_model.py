from keras.models import load_model
import keras.backend as K
import cv2
import numpy as np
import pickle
import os


def loadmodel(model_path):
    K.clear_session()
    with open(model_path, 'rb') as pickle_file:
        model = pickle.load(pickle_file)
    return model


def getvaliddatagen(validata_path):
    with open(validata_path, 'rb') as pickle_file:
        validdatagen = pickle.load(pickle_file)
    return validdatagen


def predict_images(image,model,validdatagen):
    image = np.expand_dims(image, axis=-1).astype(np.float32)/255.0
    results = model.predict_generator(
        validdatagen.flow(image,batch_size=len(image),shuffle=False),
        steps=1
    )
    y_pred = np.argmax(results, axis=-1)
    print(y_pred)
    return y_pred


if __name__ == '__main__':

    model = loadmodel('pretrained/model.pkl')
    print(model.summary())