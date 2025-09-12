import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator


model = load_model("../Modelli/line2_opt.keras")  
datagen = ImageDataGenerator()
valDat = datagen.flow_from_directory(
    "../Dataset/test",
    target_size=(48, 48),
    color_mode="grayscale",
    batch_size=64,
    class_mode="categorical",
    shuffle=False
)
print("Modello e dataset caricati")

# Importo true labels e predizioni
yTr = np.load("../Modelli/guess/trueG.npy")
yPr = np.argmax(np.load("../Modelli/guess/line2_opt.npy"), axis = 1)
cls = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Creo dataframe
appo = classification_report(yTr, yPr, target_names=cls, output_dict=True)
df = pd.DataFrame(appo).transpose()
df.to_csv("perf_line2.csv", float_format="%.4f")
