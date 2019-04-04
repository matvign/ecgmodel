from tkinter import *
from tkinter import ttk


root = Tk()
root.option_add('*tearOff', FALSE)
root.title('Synthetic ECG Generator')

menubar = Menu(root)

filemenu = Menu(menubar)
filemenu.add_command(label='Exit', command=root.quit)
menubar.add_cascade(label='File', menu=filemenu)

aboutmenu = Menu(menubar)
aboutmenu.add_command(label='About')
menubar.add_cascade(label='About', menu=aboutmenu)

root.config(menu=menubar)

root.mainloop()