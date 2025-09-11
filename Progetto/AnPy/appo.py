import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
modN = 1
testDat = 0

mod = load_model(modN)
mod.summary()
img, lab = next(testDat)
_ = mod.predict(img)

# Lista con output tensor layer convoluzionali
# e modello x attivazioni
layerOut = [
    layer.output for layer in mod.layers if 'conv' in layer.name
    ]
actMod = Model(inputs=mod.inputs, outputs=layerOut)

# Ottengo attivazioni. Valutato tutto il grafo, 
# non perdo alcuna informazione
im = img[0]
act = actMod.predict(np.expand_dims(im, axis=0))