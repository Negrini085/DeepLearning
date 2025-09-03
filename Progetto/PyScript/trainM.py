#-----------------------------------------------------#
#           Codice per allenare rete neurale          #
#-----------------------------------------------------#  

import numpy as np

from tensorflow import keras
from keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Flatten, Dense, MaxPooling2D, Conv2D, Dropout



# Funzione per la costruzione del modello
def buildMod(imW, imH, numcl):

    model = Sequential()
    model.add(Input(shape=(imW, imH, 3)))

    # Primo blocco convoluzionale
    model.add(Conv2D(16, 3, padding='same', activation='relu'))
    model.add(Conv2D(16, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    # Secondo blocco convoluzionale
    model.add(Conv2D(32, 3, padding='same', activation='relu'))
    model.add(Conv2D(32, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    # Terzo blocco convoluzionale
    model.add(Conv2D(64, 3, padding='same', activation='relu'))
    model.add(Conv2D(64, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    # Parte di classificazione
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(numcl, activation = 'softmax'))

    return model





if __name__ == "__main__":

    # Build del modello
    model = buildMod(48, 48, 7)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 


    # Loading delle immagini
    trDir = "../Dataset/train"  
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    # Dataset di training (80%)
    trDat = datagen.flow_from_directory(
        trDir,
        target_size=(48, 48),
        batch_size=32,
        class_mode="categorical",
        subset="training",     
        shuffle=True
    )

    # Dataset di validazione (20%)
    valDat = datagen.flow_from_directory(
        trDir,
        target_size=(48, 48),
        batch_size=32,
        class_mode="categorical",
        subset="validation",   
        shuffle=False
    )


    # Allenamento del modello
    earlyS = EarlyStopping(monitor='val_loss', patience=5)
    hist = model.fit(trDat, epochs=200, validation_data=valDat, callbacks=[earlyS])


    # Salvataggio quantità importanti allenamento
    loss = hist.history['loss']
    valLoss = hist.history['val_loss']
    acc = hist.history['accuracy']
    valAcc = hist.history['val_accuracy']

    data = np.column_stack((loss, valLoss, acc, valAcc))
    np.savetxt("../Modelli/training/emo_v1.dat", data, delimiter="  ", fmt="%.2f")
    model.save("../Modelli/emo_v1.keras")
