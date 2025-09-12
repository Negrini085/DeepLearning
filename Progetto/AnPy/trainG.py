import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Carico dataset
nomeF = input("Nome file da caricare: ")
df = pd.read_csv("../Modelli/training/" + nomeF)
l = len(df['loss'])+1

# Loss
fig, ax = plt.subplots(1, 2, figsize = (8, 12))
ax[0].plot(np.arange(1, l), df['loss'], color = 'darkblue', label = 'Training')
ax[0].plot(np.arange(1, l), df['val_loss'], color = 'darkred', label = 'Validation')
ax[0].set_ylabel("Loss", fontsize = 20)
ax[0].legend(loc="best", fontsize = 15)
ax[0].grid("True")

# Accuracy
ax[1].plot(np.arange(1, l), df['accuracy'], color = 'darkblue', label = 'Training')
ax[1].plot(np.arange(1, l), df['val_accuracy'], color = 'darkred', label = 'Validation')
ax[1].set_xlabel("Epoca", fontsize = 20)
ax[1].set_ylabel("Accuracy", fontsize = 20)
ax[1].legend(loc="best", fontsize = 15)
ax[1].grid("True")

plt.tight_layout()
plt.savefig("train_"+ nomeF.split("_")[0] +".png")
