#!/bin/bash

modN=("line1_v1.keras" "line1_v2.keras" "line2_v1.keras" "line2_v2.keras")      	             # Nome del file di input modello
modG=("line1_v1.npy" "line1_v2.npy" "line2_v1.npy" "line2_v2.npy")      	                     # Nome del file di output previsioni

#-----------------------------------------------------------------------#
#       Ciclo per modificare parametri esecuzione script guess.py       #
#-----------------------------------------------------------------------#
for j in "${!modN[@]}"; do
        python3 guess.py ${modN[$j]} ${modG[$j]}
done

echo "Determinazione delle guess terminata!"
