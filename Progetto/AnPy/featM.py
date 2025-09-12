import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator



# Funzione per estrazione delle feature maps
def featureM(act, nL, nameF, nC=8):

    # Dati per visualizzazione
    numF = act.shape[-1]
    size = act.shape[1]
    nR = numF // nC
    grid = np.zeros((size * nR, size * nC))
    
    for row in range(nR):
        for col in range(nC):
            # Importo l'ativazione del modello
            chImg = act[0, :, :, row * nC + col]

            # Normalizzazione (alla fine media nulla e deviazione std uno)
            chImg -= chImg.mean()
            chImg /= (chImg.std() + 1e-5)

            # Aumento contrasto e riporto in range [0, 255]
            chImg *= 64
            chImg += 128

            # Limito fra [0, 255] e inserisco immagine in griglia
            chImg = np.clip(chImg, 0, 255).astype('uint8')
            grid[row*size:(row+1)*size, col*size:(col+1)*size] = chImg
    
    if nL == 1:
        plt.imshow(act[0,:,:,7], cmap="viridis")
        plt.colorbar()
        plt.savefig("prova.png")

    scale = 1. / size
    plt.figure(figsize=(scale * grid.shape[1], scale * grid.shape[0]))
    plt.title("Feature Maps: Conv " + str(nL))
    plt.imshow(grid, aspect='auto', cmap='viridis')
    plt.axis("off")
    plt.savefig(nameF)





if __name__ == "__main__":

    #---------------------------------------------#
    #      Carico dataset + importo modello       #
    #---------------------------------------------#
    datagen = ImageDataGenerator()
    testDat = datagen.flow_from_directory(
        "../Dataset/test",
        target_size=(48, 48),
        color_mode="grayscale",
        batch_size=64,
        class_mode="categorical",
        shuffle=False
    )

    modN = input("Nome del modello da importare: ")
    model = load_model(modN)
    model.summary()
    img, lab = next(testDat)
    _ = model.predict(img)



    #-------------------------------------------------#
    #       Modello per estrazione feature maps       #
    #-------------------------------------------------#
    layerOut = [layer.output for layer in model.layers if 'conv' in layer.name]
    actMod = Model(inputs=model.inputs, outputs=layerOut)



    #-------------------------------------------------#
    #            Ottengo le attivazioni               #
    #-------------------------------------------------#
    im = img[0]
    act = actMod.predict(np.expand_dims(im, axis=0))


    #-------------------------------------------------#
    #              Stampo feature maps                #
    #-------------------------------------------------#
    featureM(act[0], 1, "feaM_line3_lay1.png")
    featureM(act[1], 2, "feaM_line3_lay2.png")
    featureM(act[2], 3, "feaM_line3_lay3.png")
    featureM(act[3], 4, "feaM_line3_lay4.png")
    featureM(act[4], 5, "feaM_line3_lay5.png")
    featureM(act[5], 6, "feaM_line3_lay6.png")
