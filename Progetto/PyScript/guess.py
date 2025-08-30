import time
import cv2 as cv
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt

from Appo.visualInput import visIn
from tensorflow.keras.models import load_model
from multiprocessing import Process, Value, Array


# Funzione per fare guess emozioni
def guessImg(numGuess, frGuess):
    
    # Come prima cosa importo il modello, perchè mi servirà per 
    # fare le guess necessarie
    model = load_model("../Modelli/dropoutTry.keras")  
    print("Modello correttamente caricato", flush = 'True')
    appo = 0

    # Creo la figura per il plot interattivo
    plt.ion()
    fig, axs = plt.subplots(1, 2, figsize=(8,4))
    axs[0].axis('off')
    fig.show()

    while True:
        if numGuess.value != appo:
            # Leggo dalle variabili globali condivise
            appo = numGuess.value
            img = np.array(frGuess[:]).reshape(48, 48)
           
            # Provo a effettuare la predizione
            cls = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
            pred = model.predict(np.expand_dims(img, axis=0))

            # Aggiorno il primo plot
            axs[0].cla()
            axs[0].imshow(img)
            axs[0].set_title("Immagine da videocamera")
            
            # Aggiorno il secondo plot
            axs[1].cla()
            pieLab = [f"{cls[i]} ({pred[0][i]*100:.1f}%)" for i in range(len(cls))]
            fette, texts = axs[1].pie(pred[0], startangle=90)                   
            axs[1].legend(fette, pieLab,                                        
                          title="Emozioni",                                     
                          loc="center left",
                          bbox_to_anchor=(1, 0, 0.5, 1))
            axs[1].set_title("Predizione: " + cls[np.argmax(pred)])
            fig.canvas.draw()
            plt.pause(0.001)



if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    # Variabili globali    
    numGuess = Value('i', 0)
    frGuess = Array('d', np.zeros(48*48))

    a = Process(target=guessImg, args=(numGuess, frGuess,), daemon=True)
    a.start()

    # Inizio la cattura video. Il parametro specificato nella funzione VideoCapture è 0 
    # perchè vogliamo che la sorgente delle immagini sia la videocamera del computer
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    durR, numG = visIn()
    start = time.perf_counter()
    lastPred = start
    # Continuo a leggere le immagini e sulle stesse effettuo le dovute analisi, come per 
    # esempio la riduzione al formato di nostro interesse (48 pixels x 48 pixels)  
    while time.perf_counter() - start <= durR:
        # Inizio a leggere lo stream frame by frame utilizzando .read() di openCV. Tale funzione 
        # restituisce un bool (True se la lettura dell'immagine è andata a buon fine, False altrimenti)
        # ed il frame in questione (che sarà quello su cui andremo ad agire).
        ret, frame = cap.read()

        # Check su lettura adeguata o meno del frame
        if not ret:
            print("Can't receive frame. Exiting ...")
            break

        # Di seguito riportiamo l'immagine catturata nel formato richiesto come input dalla 
        # rete neurale. In farticolare con cv.cvtColor la riproponiamo in scala di grigi, 
        # mentre con i due passaggi successivi selezioniamo solamente la regione centrale 
        # del frame, per poi farne un resize
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = gray[80:400, 160:480]
        gray = cv.resize(gray, (48, 48))

        # Mostriamo ora il frame a video. Per far sì che l'immagine sia visualizzata con una dimensione
        # adeguata, definiamo una finestra della quale specifichiamo la dimensione
        cv.namedWindow('frame', cv.WINDOW_NORMAL)
        cv.resizeWindow('frame', 400, 300)
        cv.imshow('frame', gray)
        if cv.waitKey(1) == ord('q'):
            break        

        # Ogni mezzo secondo aggiorno variabile condivisa, in modo tale da 
        # effettuare la predizione su un altro processo
        if time.perf_counter() - lastPred > durR/numG:
            lastPred = time.perf_counter()
            numGuess.value += 1
            frGuess[:] = gray.flatten()

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()
    a.terminate()
