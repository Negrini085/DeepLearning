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
model = load_model("../Modelli/line3_opt2.keras")
print("Dataset e modello caricati")

# Normalizzo
imB = img[0]   
imE = img[1:400]


#----------------------------------------------#
#           SHAP per feature importance        #
#----------------------------------------------#

# Il masker nasconde alcune regioni dell'immagine, per valutarne l'importanza o meno
# durante la classificazione dell'immagine
masker = shap.maskers.Image("inpaint_telea", imB.shape)

# L'explainer confronta il modello con e senza features e ne stima il contributo
# assegnado un valore shap che evidenzia il contributo positivo o negativo
cls = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
explainer = shap.Explainer(f, masker, output_names=cls)
shapVal = explainer(
    imE, max_evals=1000, batch_size=50
)


for cl in cls:
    clId = cls.index(cl)
    shapVCl = shapVal.values[:, :, :, :, clId]
    shapM = np.mean(shapVCl, axis=0) 

    plt.imshow(shapM, cmap='hot')
    plt.colorbar()
    plt.title(cl)
    plt.savefig(cl + "_hL3.png")
    plt.close()
