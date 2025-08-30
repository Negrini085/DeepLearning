import time
import cv2 as cv
import numpy as np
import multiprocessing as mp

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
    while True:
        if numGuess.value != appo:

            appo = numGuess.value
            img = np.array(frGuess[:]).reshape(48, 48)
            # Provo a fare la guess. Dato che è necessario del tempo tecnico per far sì che la 
            # rete effettui la predizione, decidiamo di provarci ogni 0.5 secondi, in modo da non
            # rallentare eccessivamente la visualizzazione
            cls = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
            pred = model.predict(np.expand_dims(img, axis=0))
            print("Predizione: " + cls[np.argmax(pred)], flush = 'True')





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
