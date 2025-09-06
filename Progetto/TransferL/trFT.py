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
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


# Variabili globali
batS = 64
imS = (128, 128)
nCl = 7





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
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )
    model.summary()

    earlyS = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    reduceLR = ReduceLROnPlateau(monitor='val_loss', factor = 0.5, patience=5, verbose=1, min_lr=1e-6)
    histo = model.fit( trDat, validation_data=valDat, epochs=50, callbacks=[earlyS, reduceLR])


    #--------------------------------------------#
    #      Salvataggio modello e history         #
    #--------------------------------------------#
    histoFT = histo.history 
    df = pd.DataFrame(histoFT)
    df.to_csv("train/FT_histo.csv", index=False)

    model.save("mod/FT.keras")
