import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Conv2D, Input, MaxPooling2D, Flatten, Dropout, Rescaling


def buildMod(imW, imH, numcl):
    model = tf.keras.models.Sequential()
    model.add(Input(shape=(imW, imH, 1)))
    model.add(Rescaling(1/255.))

    # Primo blocco convoluzionale
    model.add(Conv2D(16, 3, 
                     padding="same", activation="relu"))
    model.add(MaxPooling2D())

    # Secondo blocco convoluzionale
    model.add(Conv2D(32, 3, 
                     padding="same", activation="relu"))
    model.add(MaxPooling2D())

    # Terzo blocco convoluzionale
    model.add(Conv2D(64, 3, 
                     padding="same", activation="relu"))
    model.add(MaxPooling2D())

    # Parte di classificazione
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
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
    model.compile(optimizer="Adam", loss="categorical_crossentropy" , metrics=["accuracy"])
    model.summary()

    earlyS = EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True)
    histo = model.fit( trDat, validation_data=valDat, epochs=200, callbacks=[earlyS])



    #--------------------------------------------#
    #       Salvataggio modello e history        #
    #--------------------------------------------#

    hist = histo.history
    df = pd.DataFrame(hist)
    df.to_csv("Modelli/training/try1_histo.csv", index=False)

    model.save("Modelli/try1.keras")
