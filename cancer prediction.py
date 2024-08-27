import datetime
import tkinter as tk
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression






##tkinter interface







window=tk.Tk()
window.geometry("620x780")
window.title(" Cancer Prediction App ")
radius_mean = tk.Label(text = "radius_mean")
radius_mean.grid(column=0,row=1)
texture_mean  = tk.Label(text = "texture_mean ")
texture_mean .grid(column=0,row=2)
perimeter_mean = tk.Label(text = "perimeter_mean")
perimeter_mean.grid(column=0,row=3)
area_mean = tk.Label(text = "area_mean")
area_mean.grid(column=0,row=4)
smoothness_mean = tk.Label(text = "smoothness_mean ")
smoothness_mean .grid(column=0,row=5)
compactness_mean= tk.Label(text = "compactness_mean   ")
compactness_mean.grid(column=0,row=6)
concavity_mean = tk.Label(text = "concavity_mean")
concavity_mean.grid(column=0,row=7)
concavepoints_mean = tk.Label(text = "concave points_mean")
concavepoints_mean.grid(column=0,row=8)
symmetry_mean= tk.Label(text = "symmetry_mean")
symmetry_mean.grid(column=0,row=9)
fractal_dimension_mean = tk.Label(text = "fractal_dimension_mean")
fractal_dimension_mean.grid(column=0,row=10)
#text area
radius_meanEntry = tk.Entry()
radius_meanEntry.grid(column=1,row=1)
texture_meanEntry = tk.Entry()
texture_meanEntry.grid(column=1,row=2)
perimeter_meanEntry = tk.Entry()
perimeter_meanEntry.grid(column=1,row=3)
area_meanEntry = tk.Entry()
area_meanEntry.grid(column=1,row=4)
smoothness_meanEntry = tk.Entry()
smoothness_meanEntry.grid(column=1,row=5)
compactness_meanEntry = tk.Entry()
compactness_meanEntry.grid(column=1,row=6)
concavity_meanEntry = tk.Entry()
concavity_meanEntry.grid(column=1,row=7)
concavepoints_meanEntry = tk.Entry()
concavepoints_meanEntry.grid(column=1,row=8)
symmetry_meanEntry = tk.Entry()
symmetry_meanEntry.grid(column=1,row=9)
fractal_dimension_meanEntry = tk.Entry()
fractal_dimension_meanEntry.grid(column=1,row=10)
def getInput():
    rm=radius_meanEntry.get()
    tm=texture_meanEntry.get()
    pm=perimeter_meanEntry.get()
    am=area_meanEntry.get()
    sm=smoothness_meanEntry.get()
    cm=compactness_meanEntry.get()
    con=concavity_meanEntry.get()
    con1=concavepoints_meanEntry.get()
    sm1=symmetry_meanEntry.get()
    fm=fractal_dimension_meanEntry.get()

    
     
    df=pd.read_csv('data (1).csv')
    df=df.fillna(0)

    df['diagnosis'] = pd.to_numeric(df['diagnosis'], errors='coerce')

    df = df.dropna(subset=['diagnosis'])
    
    Diagnosis={'M':1,'B':0}
    df['diagnosis']=[Diagnosis[item] for item in df.diagnosis]

    X = df.drop(columns='diagnosis', axis=1)
    Y = df['diagnosis']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    model=LogisticRegression()
    model.fit(X_train, Y_train)
    
    input_data=(rm,tm,pm,am,sm,cm,con,con1,sm1,fm)
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = model.predict(input_data_reshaped)
    

    if (prediction[0] == 0):
        textArea = tk.Text(master=window,height=10,width=25)
        textArea.grid(column=1,row=6)
        answer = " The Breast cancer is Malignant "
        textArea.insert(tk.END,answer)

    else:
       textArea = tk.Text(master=window,height=10,width=25)
       textArea.grid(column=1,row=6)
       answer = " The Breast Cancer is Benign"
       textArea.insert(tk.END,answer)
 
    
    
    
button=tk.Button(window,text="Calculate",command=getInput,bg="pink")
button.grid(column=1,row=13) 
window.mainloop()


