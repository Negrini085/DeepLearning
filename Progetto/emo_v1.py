import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from keras import Sequential
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Flatten, Dense, MaxPooling2D, Conv2D, Dropout, RandomFlip, RandomRotation, RandomTranslation, RandomZoom, RandomContrast


# Funzione per la costruzione del modello
def buildMod(imW, imH, numcl):

    # Input e data augmentation
    model = Sequential()
    model.add(Input(shape=(imW, imH, 1)))
    model.add(RandomFlip("horizontal"))
    model.add(RandomRotation(0.1))
    model.add(RandomTranslation(0.1, 0.1))  
    model.add(RandomZoom(0.1))
    model.add(RandomContrast(0.1)) 
    
    # Primo blocco convoluzionale
    model.add(Conv2D(16, 3, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.1))
    
    # Secondo blocco convoluzionale
    model.add(Conv2D(32, 3, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.2))
    
    # Terzo blocco convoluzionale
    model.add(Conv2D(64, 3, activation='relu', padding='same'))
    model.add(Conv2D(64, 3, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.3))
    
    # Parte di classificazione
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(numcl, activation='softmax'))
   
    return model


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
    hist = model.fit(trDat, epochs=200, validation_data=valDat, callbacks=[earlyS])


    #---------------------------------------------#
    #            Salvataggio variabili            #
    #---------------------------------------------#
    histoD = hist.history 
    df = pd.DataFrame(histoD)
    df.to_csv("Modelli/training/V1_histo.csv", index=False)

    model.save("Modelli/emo_initial.keras")
