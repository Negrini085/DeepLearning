#----------------------------------------------------#
#            Codice per Transfer Learning            #
#----------------------------------------------------#
#   L'idea alla base di questo codice è quella di    #
#	utilizzare una rete pre-trained, tipo MobNetV3   #
# 	come feature extractor, modificando invece la 	 #
#	parte di classificazione per adattarla alla 	 #
#	task di riconoscimento delle emozioni. MobNetV3  #
#	è una CNN molto profonda, che evita il degrado	 #
#	delle prestazioni utilizzando "residual blocks"	 #
#----------------------------------------------------#

import os
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from keras import Sequential, Model
from keras.callbacks import EarlyStopping
from keras.applications import MobileNetV3Large
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import RandomFlip, RandomRotation, RandomTranslation, RandomZoom, RandomContrast, GlobalAveragePooling2D, Dropout, Dense


# Variabili globali
batS = 32
imS = (96, 96)
nCl = 7


# Funzione per fare strato di data augmentation. Prendiamo in particolare in considerazione quelle
# che possono essere le casistiche tipiche che si presentano per espressioni visive. Infatti vengono 
# introdotte rotazioni random (volti leggermente inclinati), traslazioni random (volti non esattamente
# centrati), zoom random e cambiamenti nel constrasto fra i vari pixels
def dataAug(rRot, rTr, rZm, rC):
    appo = tf.keras.Sequential([
        RandomFlip("horizontal"),
        RandomRotation(rRot),
        RandomTranslation(rTr, rTr),  
        RandomZoom(rZm),
        RandomContrast(rC) 
    ])
    return appo


# Funzione per la costruzione del modello basato sulla rete MobileNetV3Large
def buildMod(bMod):

    inputs = tf.keras.Input(shape=imS + (3,))
    x =  dataAug(0.1, 0.1, 0.2, 0.1)(inputs, training=True)         # Data augmentation layer
    x = tf.keras.applications.mobilenet_v3.preprocess_input(x)      # Normalizzazione immagini per MobileNetV3Large
    x = bMod(x, training=False)                                     # MobileNetV3Large
    x = GlobalAveragePooling2D()(x)                                 # Alternativo a flattening (tengo solo "forza media del layer")
    x = Dropout(0.2)(x)
    outputs = Dense(nCl, activation='softmax')(x)

    return Model(inputs, outputs)





if __name__ == "__main__":

    #---------------------------------#
    #       Caricamento dataset       #
    #---------------------------------#
    trDat = tf.keras.utils.image_dataset_from_directory(
        "../Dataset/train",
        shuffle = True,
        validation_split=0.2,
        subset="training",
        seed = 123,
        image_size=imS,
        batch_size=batS,
        color_mode="rgb",
        label_mode="int"
    )

    valDat = tf.keras.utils.image_dataset_from_directory(
        "../Dataset/train",
        shuffle = False,
        validation_split=0.2,
        subset="validation",
        image_size=imS,
        batch_size=batS,
        color_mode="rgb",
        label_mode="int"
    )
    print("Dataset correttamente caricato!")



    #----------------------------------------------------#
    #       Caricamento modello MobileNetV3Large         #
    #----------------------------------------------------#
    # Importo il modello RedNet18 non includendo il layer di classificazione, fissando inoltre 
    # i parametri in modo che non possano essere modificati durante l'allenamento
    bMod = MobileNetV3Large(input_shape=imS + (3,), weights="imagenet", include_top=False)
    bMod.trainable = False



    #--------------------------------------------#
    #       Costruzione modello transfer         #
    #--------------------------------------------#
    model = buildMod(bMod)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )
    model.summary()


    #--------------------------------------------#
    #           Prefetch & Allenamento           #
    #--------------------------------------------#
    AUTOTUNE = tf.data.AUTOTUNE
    trDat = trDat.prefetch(buffer_size=AUTOTUNE)
    valDat = valDat.prefetch(buffer_size=AUTOTUNE)

    earlyS = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    histo = model.fit(trDat, validation_data=valDat, epochs=200, callbacks=[earlyS])


    #--------------------------------------------#
    #      Salvataggio modello e history         #
    #--------------------------------------------#
    histoD = histo.history 
    df = pd.DataFrame(histoD)
    df.to_csv("train/V1_histo.csv", index=False)

    model.save("mod/V1.keras")
