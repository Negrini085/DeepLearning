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
from keras.applications import MobileNetV3Small
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import RandomFlip, RandomRotation, RandomTranslation, RandomZoom, RandomContrast, GlobalAveragePooling2D, Dropout, Dense


# Variabili globali
batS = 32
imS = (128, 128)
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


# Funzione per la costruzione del modello basato sulla rete MobileNetV3Small
def buildMod(bMod):

    inputs = tf.keras.Input(shape=imS + (3,))
    x =  dataAug(0.1, 0.1, 0.2, 0.1)(inputs, training=True)         # Data augmentation layer
    x = tf.keras.applications.mobilenet_v3.preprocess_input(x)      # Normalizzazione immagini per MobileNetV3Small
    x = bMod(x, training=False)                                     # MobileNetV3Small
    x = GlobalAveragePooling2D()(x)                                 # Alternativo a flattening (tengo solo "forza media del layer")
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(nCl, activation='softmax')(x)

    return Model(inputs, outputs)





if __name__ == "__main__":

    #---------------------------------#
    #       Caricamento dataset       #
    #---------------------------------#
    print("Inizia la carica del dataset")
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


    #--------------------------------------------#
    #                   Prefetch                 #
    #--------------------------------------------#
    AUTOTUNE = tf.data.AUTOTUNE
    trDat = trDat.prefetch(buffer_size=AUTOTUNE)
    valDat = valDat.prefetch(buffer_size=AUTOTUNE)


    #--------------------------------------------#
    #           Fine-tuning MobileNetV3          #
    #--------------------------------------------#
    model = tf.keras.models.load_model("mod/V1.keras")
    print("Modello correttamente caricato!")

    bMod = model.layers[2]
    bMod.trainable = True

    # Mantiengo congelati l'80% dei layer
    fine_tune_at = int(len(bMod.layers) * 0.8)
    for layer in bMod.layers[:fine_tune_at]:
        layer.trainable = False


    #--------------------------------------------#
    #             Allenamento rete               #
    #--------------------------------------------#
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )
    model.summary()

    earlyS = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduceLR = ReduceLROnPlateau(monitor='val_loss', factor = 0.2, patience=3, verbose=1, min_lr=1e-6)
    histo = model.fit( trDat, validation_data=valDat, epochs=50, callbacks=[earlyS, reduceLR])


    #--------------------------------------------#
    #      Salvataggio modello e history         #
    #--------------------------------------------#
    histoFT = histo.history 
    df = pd.DataFrame(histoFT)
    df.to_csv("train/FT_histo.csv", index=False)

    model.save("mod/FT.keras")
