#---------------------------------------------------------------#
#       Codice per determinare le guess di un dato modello      #
#---------------------------------------------------------------#
import sys

# Controllo quanti argomenti sono presenti, poichè in caso ne siano
# stati forniti troppi pochi, non continuo con l'esecuzione del codice
if len(sys.argv) == 3:
        
    import numpy as np

    from tensorflow import keras
    from tensorflow.keras.models import load_model, Model
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

    # Funzione che effettua le guess per un certo modello e stampa il risultato a file,
    # di modo che non sia necessario ri-eseguire quanto già fatto una volta
    def guessMod(modN, valDir, title):

        # Importo il modello e le immagini sulle quali effettuare le guess
        model = load_model(modN)  
        
        datagen = ImageDataGenerator()
        valDat = datagen.flow_from_directory(
            valDir,
            target_size=(48, 48),
            color_mode="grayscale",
            batch_size=64,
            class_mode="categorical",
            shuffle=False
        )

        # Predizioni del modello
        pred = model.predict(valDat)
        np.save(title, pred)


    if __name__ == "__main__":
        
        # Lettura argomenti ulteriori
        modN = sys.argv[1]
        title = sys.argv[2]
        
        guessMod("../Modelli/" + modN, "../Dataset/test", "../Modelli/guess/" + title)
        print("Guess stampate a file!")
        
else:
    print("Utilizzo corretto del programma: python3 nome_file <nome modello> <titolo file output>")
