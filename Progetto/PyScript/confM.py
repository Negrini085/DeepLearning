#---------------------------------------------------------------#
#      Codice per fare la confusion map di un dato modello      #
#---------------------------------------------------------------#
import sys

# Controllo quanti argomenti sono presenti, poichè in caso ne siano
# stati forniti troppi pochi, non continuo con l'esecuzione del codice
if len(sys.argv) == 3:
        
    import itertools
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    from sklearn.metrics import confusion_matrix

    # Funzione che consente di stampare la matrice di confusione per un certo modello
    # dati nome del file delle guess 
    def confMat(modN, titM):

        # Creazione matrice di confusione
        yTr = np.load("../Modelli/guess/trueG.npy")
        yPr = np.load(modN)

        cm = confusion_matrix(yTr, yPr, normalize = 'true')
        labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        title='Confusion matrix: ' + titM

        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, round(cm[i, j], 2),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('Vera label')
        plt.xlabel('Guess label')
        plt.tight_layout()
        plt.savefig("../Modelli/immagini/confM_" + titM + ".png")


    if __name__ == "__main__":
        
        # Lettura argomenti ulteriori
        modN = sys.argv[1]
        titM = sys.argv[2]

        confMat("../Modelli/guess/" + modN, titM)
        print("Matrice di confusione stampata a file!")
        
else:
    print("Utilizzo corretto del programma: python3 nome_file <nome modello> <titolo file output>")