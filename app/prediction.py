import tensorflow
from tensorflow import keras
from keras.models import load_model
from keras.utils import img_to_array
import cv2 as cv
import numpy as np
import pickle
import logging



with open("model/model_labels.pkl", "rb") as labels:
    model_labels=pickle.load(labels)
    
model=load_model("model/cnnModel.h5")

def preprocessImage(image):
        try:
            imageArray=img_to_array(image)
            imageArray=imageArray/255
            imageArray=np.expand_dims(imageArray, axis=0)
            return imageArray
        except Exception as e:
            logging.exception(e)
        
def makePrediction(image):
    try:        
        image=cv.resize(image, (224, 224))
        imageArray=preprocessImage(image=image)        
        prediction=model.predict(imageArray)        
        confidence=round(100 * (np.max(prediction)), 2)        
        prediction=model_labels[np.argmax(prediction)]  
        print(prediction)              
        return prediction, confidence
    except Exception as e:
        logging.exception(e)
        



    
