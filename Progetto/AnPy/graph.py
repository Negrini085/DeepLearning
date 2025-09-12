import pandas as pd
import matplotlib.pyplot as plt



# Carico report e stampo colonna di nostro interesse
df = pd.read_csv("perf_line3.csv")
cls = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


# Precision
fig = plt.figure(figsize=(12, 6))
precision = df["precision"]
plt.bar(range(7), precision, color='skyblue', edgecolor='black')
plt.xticks(range(7), [i for i in cls], fontsize = 15)
plt.ylabel("Precision", fontsize = 15)
plt.title("Linea 2", fontsize = 20)
plt.savefig("prec_line3.png")
print(precision)

# Recall
fig = plt.figure(figsize=(12, 6))
recall = df["recall"]
plt.bar(range(7), recall, color='skyblue', edgecolor='black')
plt.xticks(range(7), [i for i in cls], fontsize = 15)
plt.ylabel("Recall", fontsize = 15)
plt.title("Linea 2", fontsize = 20)
plt.savefig("rec_line3.png")
print(recall)

# F1-score
fig = plt.figure(figsize=(12, 6))
f1_score = df["f1-score"]
plt.bar(range(7), f1_score, color='skyblue', edgecolor='black')
plt.xticks(range(7), [i for i in cls], fontsize = 15)
plt.ylabel("F1 score", fontsize = 15)
plt.title("Linea 2", fontsize = 20)
plt.savefig("f1_line3.png")
print(f1_score)
