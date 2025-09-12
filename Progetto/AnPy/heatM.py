import shap
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


# Funzione modello compatibile con shap (non uso predict perchè così
# facendo ho un tensore ed il calcolo dei gradienti è comodo)
def f(x):
    return model(x)

#-------------------------------------------#
#         Carico dataset e modello          #
#-------------------------------------------#
trDat = tf.keras.utils.image_dataset_from_directory(
    "../Dataset/train",
    shuffle=True,
    seed=123,
    image_size=(48, 48),
    batch_size=128,
    color_mode="grayscale",
    label_mode="categorical"
)

img = []
lab = []

# Itero sui primi 15 batch
for i, (im, l) in enumerate(trDat):
    img.append(im.numpy())
    lab.append(l.numpy())
    if i == 14: 
        break

print()
img = np.concatenate(img, axis=0)
lab = np.concatenate(lab, axis=0)
model = load_model("../Modelli/loadable/line2_opt.keras")
print("Dataset e modello caricati")

# Normalizzo
imB = img[0]   
imE = img[1:3]

#----------------------------------------------#
#           SHAP per feature importance        #
#----------------------------------------------#
cls = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
shap_values_array = np.load("shap_line2.npy", allow_pickle=True)
shap_values = shap.Explanation(values=shap_values_array, data=imE, output_names=cls)
shap.image_plot(shap_values.values[:3], shap_values.data[:3])
