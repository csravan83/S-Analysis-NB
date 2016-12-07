from Tkinter import *
import ttk
import nbclassify
root = Tk()
def on_select(*args):
    selected =  productvar.get()
    val = False
    if selected == 'iPhone6s':
        val = nbclassify.init('testdata.csv')
    else:
        val = nbclassify.init('testdata2.csv')
    if val:
        label = Label(root, text="The Product has got many Positive reviews. Hence the General Sentiment is positive")
        label.pack()
    else:
        label = Label(root, text="The Product has got many Negative reviews. Hence the General Sentiment is positive")
        label.pack()

root.wm_title("Product Reviews")

productvar = StringVar()
products = ttk.Combobox(root, textvariable=productvar)
products['values'] = ['iPhone6s', 'Micromax Canvas']
products.pack()
products.bind('<<ComboboxSelected>>', on_select)
root.mainloop()
