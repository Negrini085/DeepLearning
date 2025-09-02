#!/bin/bash

modN=("firstTry.keras")      	             # Nome del file di input modello
modG=("firstTry.npy")      	             # Nome del file di output previsioni

#-----------------------------------------------------------------------#
#       Ciclo per modificare parametri esecuzione script guess.py       #
#-----------------------------------------------------------------------#
for j in "${!modN[@]}"; do
        python3 guess.py ${modN[$j]} ${modG[$j]}
done

echo "Determinazione delle guess terminata!"
