from tkinter import *
from tkinter import ttk

import sentim
root = Tk()

def on_select(*args):
    selected =  productvar.get()
    val = False
    if selected == 'iPhone5s':
        val = sentim.init('testdata.csv')
    else:
        val = sentim.init('testdata2.csv')
    if val:
        labelvar.set("The Product has got many Positive reviews. Hence the General Sentiment is positive")
    else:
        labelvar.set("The Product has got many Negative reviews. Hence the General Sentiment is negative")


root.wm_title("Product Reviews")
productvar = StringVar()
labelvar = StringVar()
header = Label(root, text="Welcome to Sentiment Analysis")
header.config(font=("Courier", 26), pady=30)
header.pack()
products = ttk.Combobox(root, textvariable=productvar)
products['values'] = ['iPhone5s', 'Micromax Canvas Spark3']
products.pack()
labelvar.set("Select a product to show sentiment.")
label = Label(root, textvar=labelvar)
label.config(font=("Arial",12),pady=20)
label.pack()
products.bind('<<ComboboxSelected>>', on_select)
root.minsize(width=666, height=300)
root.maxsize(width=666, height=300)
root.mainloop()
