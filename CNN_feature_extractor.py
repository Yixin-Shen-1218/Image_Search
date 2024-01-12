from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow import keras
import numpy as np


# define a feature extractor class, using it as CNN feature extractor
class FeatureExtractor:
    def __init__(self):
        # the output of model is the output of the second last layer
        base_model = keras.models.load_model('./cnn_model/')
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer("dense2").output)

    def extract(self, img):
        img = img.resize((64, 64)).convert("RGB")
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feature = self.model.predict(x)[0]
        return feature / np.linalg.norm(feature)
