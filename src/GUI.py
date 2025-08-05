import tkinter as tk
from tkinter import StringVar, OptionMenu, Text, END, W, Label, Button
import joblib
import pandas as pd
import numpy as np

# Load data and model
df = pd.read_csv('data/cleaned_file.csv')
symptoms_list = list(df.columns[:-1])
l1 = symptoms_list
disease = joblib.load('model/gaussian_nb_model.pkl').classes_
gnb = joblib.load('model/gaussian_nb_model.pkl')

def message():
    psymptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get(), Symptom5.get()]
    l2 = [0] * len(l1)

    for z in psymptoms:
        if z in l1:
            index = l1.index(z)
            l2[index] = 1

    inputtest = [l2]
    predict = gnb.predict(inputtest)
    predicted = predict[0]

    t3.delete("1.0", END)
    t3.insert(END, f"Predicted Disease: {predicted}")

# GUI Setup
root = tk.Tk()
root.title("DISEASE PREDICTION FROM SYMPTOMS")
root.configure(bg='lavender')

Symptom1 = StringVar()
Symptom1.set(None)
Symptom2 = StringVar()
Symptom2.set(None)
Symptom3 = StringVar()
Symptom3.set(None)
Symptom4 = StringVar()
Symptom4.set(None)
Symptom5 = StringVar()
Symptom5.set(None)

Label(root, text="DISEASE PREDICTION FROM SYMPTOMS", font=("Bold", 20), bg='lavender').grid(row=1, column=1, pady=10, columnspan=2)

OPTIONS = sorted(l1)

Label(root, text="Symptom 1", font=("Arial", 12), bg='lavender').grid(row=3, column=1, sticky=W, padx=10)
OptionMenu(root, Symptom1, *OPTIONS).grid(row=3, column=2)

Label(root, text="Symptom 2", font=("Arial", 12), bg='lavender').grid(row=4, column=1, sticky=W, padx=10)
OptionMenu(root, Symptom2, *OPTIONS).grid(row=4, column=2)

Label(root, text="Symptom 3", font=("Arial", 12), bg='lavender').grid(row=5, column=1, sticky=W, padx=10)
OptionMenu(root, Symptom3, *OPTIONS).grid(row=5, column=2)

Label(root, text="Symptom 4", font=("Arial", 12), bg='lavender').grid(row=6, column=1, sticky=W, padx=10)
OptionMenu(root, Symptom4, *OPTIONS).grid(row=6, column=2)

Label(root, text="Symptom 5", font=("Arial", 12), bg='lavender').grid(row=7, column=1, sticky=W, padx=10)
OptionMenu(root, Symptom5, *OPTIONS).grid(row=7, column=2)

Button(root, text="PREDICT", height=2, width=20, command=message, font=("Arial", 12), bg="light green").grid(row=8, column=1, pady=20, columnspan=2)

t3 = Text(root, height=2, width=40, font=("Arial", 14))
t3.grid(row=9, column=1, columnspan=2, pady=10)

root.mainloop()