import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import RandomRotation, RandomTranslation, RandomZoom, RandomContrast
from tensorflow.keras.layers import Dense, Conv2D, Input, MaxPooling2D, Flatten, Dropout, Rescaling, BatchNormalization, Activation

nLay = 19
drR = 0.13444894387463033

def buildMod(imW, imH, numcl):
    model = tf.keras.models.Sequential()
    model.add(Input(shape=(imW, imH, 1)))
    model.add(Rescaling(1/255.))

    # Data augmentation
    model.add(RandomRotation(0.15))
    model.add(RandomTranslation(0.1, 0.1))
    model.add(RandomZoom(0.1))
    model.add(RandomContrast(0.1))

    # Primo blocco convoluzionale
    model.add(Conv2D(nLay, 3, padding="same", activation="relu"))
    model.add(MaxPooling2D())

    # Secondo blocco convoluzionale
    model.add(Conv2D(2*nLay, 3, padding="same", activation="relu"))
    model.add(MaxPooling2D())

    # Terzo blocco convoluzionale
    model.add(Conv2D(4*nLay, 3, padding="same", activation="relu"))
    model.add(MaxPooling2D())

    # Quarto blocco convoluzionale
    model.add(Conv2D(8*nLay, 3, padding="same", activation="relu"))
    model.add(Dropout(drR))
    model.add(MaxPooling2D())

    # Parte di classificazione
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(2*drR))
    model.add(Dense(numcl, activation = "softmax"))

    return model





if __name__ == "__main__":

    #---------------------------------#
    #        Caricamento dataset      #
    #---------------------------------#
    print("Inizia la carica del dataset")
    trDat = tf.keras.utils.image_dataset_from_directory(
        "Dataset/train",
        shuffle = True,
        validation_split = 0.2,
        subset = "training",
        seed = 123,
        image_size = (48, 48),
        batch_size = 64,
        color_mode = "grayscale",
        label_mode = "categorical"
    )
    
    valDat = tf.keras.utils.image_dataset_from_directory(
        "Dataset/train",
        shuffle = True,
        validation_split = 0.2,
        subset = "validation",
        seed = 123,
        image_size = (48, 48),
        batch_size = 64,
        color_mode = "grayscale",
        label_mode = "categorical"
    )
    print("Dataset correttamente caricato!")



    #--------------------------------------------#
    #                 Prefetch                   #
    #--------------------------------------------#
    AUTOTUNE = tf.data.AUTOTUNE
    trDat = trDat.prefetch(buffer_size=AUTOTUNE)
    valDat = valDat.prefetch(buffer_size=AUTOTUNE)



    #--------------------------------------------#
    #              Build & Training              #
    #--------------------------------------------#
    model = buildMod(48, 48, 7)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss= "categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    earlyS = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    reduceLR = ReduceLROnPlateau(monitor='val_loss', factor = 0.5, patience=3, verbose=1, min_lr=1e-6)
    histo = model.fit(trDat, validation_data=valDat, epochs=200, callbacks=[earlyS, reduceLR])



    #--------------------------------------------#
    #       Salvataggio modello e history        #
    #--------------------------------------------#
    hist = histo.history
    df = pd.DataFrame(hist)
    df.to_csv("Modelli/training/line2_opt_histo.csv", index=False)

    model.save("Modelli/line2_opt.keras")
