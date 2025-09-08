from tkinter import *
from tkinter import ttk

# Funzione necessaria per la lettura dei valori una volta inseriti nella finestra Tkinter
# In particolare legge i valori inseriti nelle Entry specificate nella funzione visIn ed 
# infine distrugge la root
def conferma(root, dur, gue):
    durInt = dur.get()
    nG = gue.get()
    print("Durata inserita:", durInt)
    print("Numero di guess:", nG)
    root.destroy()


# Questa funzione serve per per creare una funzione di Input con Tkinter. In particolare, 
# una volta creata la finestra stessa, vengono richieste la durata della registrazione ed 
# il numero di guess da effettuare. Il tutto è coronato da un bottone "Conferma", che porta 
# alla chiusura della finestra stessa. La funzione visIn restituisce una tupla, costituita 
# dalla durata della registrazione e dal numero di guess.
def visIn():
    
    # Creazione finestra per visualizzazione
    root = Tk()
    root.title("Richieste di input")
    root.geometry('600x400')

    # Richiesta durata della registrazione 
    labD = Label(root, text="Durata della registrazione")
    labD.pack(padx=5, pady=5)

    dur = StringVar()
    entryD = ttk.Entry(root, textvariable=dur)
    entryD.pack(padx=5, pady=5)

    # Richiesta numero di guess da fare
    numG = Label(root, text="Numero di guess")
    numG.pack(padx=5, pady=5)

    gue = StringVar()
    entryG = ttk.Entry(root, textvariable=gue)
    entryG.pack(padx=5, pady=5)

    # Creazione del pulsante e chiusura della finestra stessa
    btn = Button(root, text="Conferma", command= lambda: conferma(root, dur, gue))
    btn.pack(pady=20)

    root.mainloop()
    return (float(dur.get()), float(gue.get()))
