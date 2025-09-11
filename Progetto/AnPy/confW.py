import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix


# Funzione per stampare i due allenamenti
def printTr(name, nameW):

    # Importo i dati
    fig, ax = plt.subplots(2, 1, figsize=(7, 12))
    df = pd.read_csv("../Modelli/training/" + name)
    dfW = pd.read_csv("../Modelli/training/" + nameW)

    # No class weights    
    l = len(df["loss"])+1
    ax[0].plot(np.arange(1, l), df["accuracy"], color = "darkblue", label = "Training")
    ax[0].plot(np.arange(1, l), df["val_accuracy"], color = "darkred", label = "Validation")
    ax[0].set_title("No class weights", fontsize = 20)
    ax[0].set_ylabel("Accuracy", fontsize = 15)
    ax[0].legend(loc="best")
    ax[0].grid(True)

    # Class weights
    l = len(dfW["loss"])+1
    ax[1].plot(np.arange(1, l), dfW["accuracy"], color = "darkblue", label = "Training")
    ax[1].plot(np.arange(1, l), dfW["val_accuracy"], color = "darkred", label = "Validation")
    ax[1].set_title("Class weights", fontsize = 20)
    ax[1].set_ylabel("Accuracy", fontsize = 15)
    ax[1].set_ylabel("Epoche", fontsize = 15)
    ax[1].legend(loc="best")
    ax[1].grid(True)

    plt.savefig("classW.png")



# Funzione per stampare le due matrici
def printMat():

    # Importo le guess
    yTr = np.load("../Modelli/guess/trueG.npy")

    fig, ax = plt.subplots(2, 1, figsize = (7, 12))
    titM = ["No class weights", "Class weights"]
    names = ["../Modelli/guess/try1.npy", "../Modelli/guess/try1W.npy"]


    for i in range(0, 2):
    
        yPr = np.argmax(np.load(names[i]), axis = 1)
        acc = (yPr == yTr).mean()
        print("Accuracy globale:", acc)

        cm = confusion_matrix(yTr, yPr, normalize = 'true')
        labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        title=titM[i]

        im = ax[i].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax[i].set_title(title, fontsize = 20)
        plt.colorbar(im, ax=ax[i])  
        tick_marks = np.arange(len(labels))
        ax[i].set_xticks(tick_marks, labels, rotation=45)
        ax[i].set_yticks(tick_marks, labels)
        thresh = cm.max() / 2.
        for k, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax[i].text(j, k, round(cm[k, j], 2),
                    horizontalalignment="center",
                    color="white" if cm[k, j] > thresh else "black")

        ax[i].set_ylabel('Vera label')
        ax[i].set_xlabel('Guess label')
    
    plt.tight_layout()
    plt.savefig("matrixW.png")




if __name__ == "__main__":
    printTr("try1_histo.csv", "try1W_histo.csv")
    printMat()