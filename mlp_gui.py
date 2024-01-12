import tkinter as tk
from tkinter import messagebox
import pandas as pd
from keras.models import load_model
import numpy as np

# Load the trained MLP model
model_filename = 'mlp_model.h5'
mlp_model = load_model(model_filename)

def make_prediction():
    try:
        # Get user input from entry widgets
        lm35_val = float(lm35_entry.get())
        dht11_val = float(dht11_entry.get())
        spo2_val = float(spo2_entry.get())
        bpm_val = float(bpm_entry.get())

        # Make prediction
        input_data = np.array([[lm35_val, dht11_val, spo2_val, bpm_val]])
        predicted_class = np.argmax(mlp_model.predict(input_data))

        # Show prediction message
        if predicted_class == 0:
            messagebox.showinfo("Prediction", "Patient is Unhealthy (Class 0)")
        elif predicted_class == 1:
            messagebox.showinfo("Prediction", "Patient is Normal (Class 1)")
        else:
            messagebox.showinfo("Prediction", "Patient is Healthy (Class 2)")

    except ValueError:
        messagebox.showerror("Error", "Please enter valid numerical values.")

# Create the main window
root = tk.Tk()
root.title("Health Prediction App")

# Create entry widgets for user input
lm35_label = tk.Label(root, text="LM35 Value:")
lm35_label.grid(row=0, column=0)
lm35_entry = tk.Entry(root)
lm35_entry.grid(row=0, column=1)

dht11_label = tk.Label(root, text="DHT11 Value:")
dht11_label.grid(row=1, column=0)
dht11_entry = tk.Entry(root)
dht11_entry.grid(row=1, column=1)

spo2_label = tk.Label(root, text="SpO2 Value:")
spo2_label.grid(row=2, column=0)
spo2_entry = tk.Entry(root)
spo2_entry.grid(row=2, column=1)

bpm_label = tk.Label(root, text="BPM Value:")
bpm_label.grid(row=3, column=0)
bpm_entry = tk.Entry(root)
bpm_entry.grid(row=3, column=1)

# Create a button to make predictions
predict_button = tk.Button(root, text="Make Prediction", command=make_prediction)
predict_button.grid(row=4, column=0, columnspan=2, pady=10)

# Start the Tkinter event loop
root.mainloop()
