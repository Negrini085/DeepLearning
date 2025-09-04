import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Flatten, Dense, MaxPooling2D, Conv2D, Dropout, RandomFlip, RandomRotation, RandomTranslation, RandomZoom, RandomContrast, Rescaling


# Funzione per data augmentation
def dataAug(rRot, rTr, rZm, rC):
    appo = tf.keras.Sequential([
        RandomFlip("horizontal"),
        RandomRotation(rRot),
        RandomTranslation(rTr, rTr),  
        RandomZoom(rZm),
        RandomContrast(rC) 
    ])
    return appo


# Funzione per la costruzione del modello
def buildMod(imW, imH, numcl):
    inputs = Input(shape=(imW, imH, 1))
    
    # Data augmentation
    x = dataAug(0.1, 0.1, 0.2, 0.1)(inputs)
    
    # Primo blocco convoluzionale
    x = Conv2D(16, 3, activation='relu', padding='same')(x)
    x = Conv2D(16, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = Dropout(0.1)(x)
    
    # Secondo blocco convoluzionale
    x = Conv2D(32, 3, activation='relu', padding='same')(x)
    x = Conv2D(32, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = Dropout(0.2)(x)
    
    # Terzo blocco convoluzionale
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(3, 3), padding='same')(x)
    x = Dropout(0.3)(x)
    
    # Parte di classificazione
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    outputs = Dense(numcl, activation='softmax')(x)
    
    return Model(inputs=inputs, outputs=outputs)


# Funzione per rescaling immagini
def rescale(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label





if __name__ == "__main__":

    #---------------------------------------------#
    #               Importo dataset               #
    #---------------------------------------------#
    trDat = tf.keras.utils.image_dataset_from_directory(
        "Dataset/train",
        shuffle = True,
        validation_split=0.2,
        subset="training",
        seed = 123,
        image_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        label_mode="categorical"
    )

    valDat = tf.keras.utils.image_dataset_from_directory(
        "Dataset/train",
        shuffle = False,
        validation_split=0.2,
        subset="validation",
        image_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        label_mode="categorical"
    )

    trDat = trDat.map(rescale).prefetch(tf.data.AUTOTUNE)
    valDat = valDat.map(rescale).prefetch(tf.data.AUTOTUNE)
    print("Dataset correttamente caricato!")


    #---------------------------------------------#
    #             Costruzione modello             #
    #---------------------------------------------#
    model = buildMod(48, 48, 7)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 
    model.summary()


    #---------------------------------------------#
    #             Allenamento modello             #
    #---------------------------------------------#
    earlyS = EarlyStopping(monitor='val_loss', patience=5)
    hist = model.fit(trDat, epochs=1, validation_data=valDat, callbacks=[earlyS])


    #---------------------------------------------#
    #            Salvataggio variabili            #
    #---------------------------------------------#
    histoD = hist.history 
    df = pd.DataFrame(histoD)
    df.to_csv("Modelli/training/V1_histo.csv", index=False)

    model.save("Modelli/emo_initial.keras")
