import optuna
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import RandomRotation, RandomTranslation, RandomZoom, RandomContrast
from tensorflow.keras.layers import Dense, Conv2D, Input, MaxPooling2D, Flatten, Dropout, Rescaling, BatchNormalization, Activation


from optuna.samplers import RandomSampler
from optuna.visualization import plot_contour , plot_rank


#-----------------------------------------------------#
#        Caricamento dataset (variabili globali)      #
#-----------------------------------------------------#
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

testDat = tf.keras.utils.image_dataset_from_directory(
    "Dataset/test",
    shuffle = True,
    seed = 123,
    image_size = (48, 48),
    batch_size = 64,
    color_mode = "grayscale",
    label_mode = "categorical"
)



#-----------------------------------------------------#
#                 Definizione funzioni                #
#-----------------------------------------------------#

# Funzione per costruire il modello
def buildMod(trial):

    # Scelgo numero filtri e dropout
    nFil = trial.suggest_int("filters", 4, 20, step=2)
    drR = trial.suggest_float("dropout", 0.1, 0.3)

    # Parte di input
    model = tf.keras.models.Sequential()
    model.add(Input(shape=(48, 48, 1)))
    model.add(Rescaling(1/255.))

    # Data augmentation
    model.add(RandomRotation(0.15))
    model.add(RandomTranslation(0.1, 0.1))
    model.add(RandomZoom(0.1))
    model.add(RandomContrast(0.1))

    # Primo blocco convoluzionale
    model.add(Conv2D(nFil, 3, padding="same", activation="relu"))
    model.add(MaxPooling2D())

    # Secondo blocco convoluzionale
    model.add(Conv2D(2*nFil, 3, padding="same", activation="relu"))
    model.add(MaxPooling2D())

    # Terzo blocco convoluzionale
    model.add(Conv2D(4*nFil, 3, padding="same", activation="relu"))
    model.add(MaxPooling2D())

    # Quarto blocco convoluzionale
    model.add(Conv2D(8*nFil, 3, padding="same", activation="relu"))
    model.add(Dropout(drR))
    model.add(MaxPooling2D())

    # Parte di classificazione
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(2*drR))
    model.add(Dense(7, activation = "softmax"))

    return model



# Funzione per allenare modello
def train(trDat, valDat, trial):
    model = buildMod(trial)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss= "categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    earlyS = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    reduceLR = ReduceLROnPlateau(monitor='val_loss', factor = 0.5, patience=3, verbose=1, min_lr=1e-6)
    model.fit(trDat, validation_data=valDat, epochs=30, callbacks=[earlyS, reduceLR])

    return model


# Funzione obiettivo (necessaria per optuna)
def objective(trial):
    model = train(trDat, valDat, trial)
    score = model.evaluate(testDat)

    return score[1]







if __name__ == "__main__":
    #--------------------------------------------#
    #                 Prefetch                   #
    #--------------------------------------------#
    AUTOTUNE = tf.data.AUTOTUNE
    trDat = trDat.prefetch(buffer_size=AUTOTUNE)
    valDat = valDat.prefetch(buffer_size=AUTOTUNE)



    #--------------------------------------------#
    #            Iper-ottimizzazione             #
    #--------------------------------------------#
    sampler = RandomSampler(seed=0)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=20)

    df = study.trials_dataframe()
    df.to_csv("line2_opt1.csv", index=False)
