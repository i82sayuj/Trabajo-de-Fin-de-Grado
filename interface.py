from tkinter import filedialog
from tkinter import *
from tkinter.ttk import*
from functools import partial
import runpy
import sys
import os

def btm(nombre,c):
        sys.argv = [' ',str(nombre.current()),c.get()]
        runpy.run_path('./build/scripts-3.9/main.py',run_name = '__main__')



def particle():
    root = Tk()
    root.title("Particle indentify")
    root.config(width="700", height="500",background = "#38b382")
    root.iconbitmap('images\\icon.ico')
    root.resizable(0,0)
    
    frame1 = Frame(root)
    frame1.place(x = 250, y = 200)

    texto =Label(frame1, text = "Seleccione la partícula a identificar")
    texto.grid(row=0,column = 0)

    f = open('files\\particles.txt', 'r')

    options = []

    for i in f:
        line = i.split(sep = ',')
        options.append(line[0])

    w = Combobox(frame1, values = options)
    w.grid(row= 1, column = 0)

    texto2 =Label(frame1, text = "Seleccione una imagen a procesar")
    texto2.grid(row = 3,column = 0)

    def OpenFile():
        archivo_abierto = StringVar()
        archivo_abierto.set(filedialog.askopenfilename(initialdir = "/", title = "Seleccione archivo", filetypes = (("jpeg files", " .jpg"),("all files","*.*"))))
        c.set(archivo_abierto.get())

    c = StringVar()
    Button(frame1,text = "Abrir archivo",command = OpenFile).grid(row = 4,column = 0)
    
    btm_arg = partial(btm,w,c)
    button = Button(frame1,text = "send",command = btm_arg)
    button.grid(row = 5, column = 0)

    root.mainloop()


def fileWrite(EntryNombre,EntryMinBtm,EntryMaxBtm,EntryMargin,root):
        
        if(EntryNombre.get() == "" or EntryMinBtm.get()=="" or EntryMaxBtm.get()=="" or EntryMargin.get() == ""):
                print("Error, debe especificar todos los campos.")
                root.quit()

        file = open('files\\particles.txt','a')
        line = EntryNombre.get()+','+EntryMinBtm.get()+','+EntryMaxBtm.get()+','+EntryMargin.get()+'\n'
        file.write(line)

        file.close()
        print("partícula añadida.")
        root.destroy()

def addParticle():

    root = Tk()
    root.title("Add Particle")
    root.config(width="700", height="500",background = "#38b382")
    root.iconbitmap('images\\icon.ico')
    root.resizable(0,0)
    
    frame1 = Frame(root)
    frame1.place(x = 200, y = 175)

    nombre = Label(frame1,text = "Nombre de la partícula: ")
    nombre.grid(row = 0,column = 0)

    EntryNombre = Entry(frame1)
    EntryNombre.grid(row= 0,column = 1,pady = 5,padx = 5)

    MinBtm = Label(frame1,text = "Btm mínimo de la partícula: ")
    MinBtm.grid(row = 1,column = 0)

    EntryMinBtm = Entry(frame1)
    EntryMinBtm.grid(row= 1,column = 1,pady = 5,padx = 5)

    MaxBtm = Label(frame1,text = "Btm máximo de la partícula: ")
    MaxBtm.grid(row = 2,column = 0)

    EntryMaxBtm = Entry(frame1)
    EntryMaxBtm.grid(row= 2,column = 1, pady = 5,padx = 5)

    margin = Label(frame1,text = "Margen de borde: ")
    margin.grid(row = 3,column = 0)

    EntryMargin = Entry(frame1)
    EntryMargin.grid(row= 3,column = 1, pady = 5,padx = 5)

    v = StringVar()
    fileWrite_arg = partial(fileWrite,EntryNombre,EntryMinBtm,EntryMaxBtm,EntryMargin,root)
    button = Button(frame1,text = "send", width = 15,command = fileWrite_arg)
    button.grid(row = 4, column = 0)


def modifyFile(nombre,EntryNombre,EntryMinBtm,EntryMaxBtm,EntryMargin,root):
        contenido= list()
        with open('files\\particles.txt', 'r') as archivo:
                for linea in archivo:
                        columnas = linea.split(',')
                        print(nombre.get())
                        print(columnas[0])
                        if(nombre.get()  == columnas[0]):
                                contenido.append(EntryNombre.get()+','+EntryMinBtm.get()+','+EntryMaxBtm.get()+','+EntryMargin.get()+'\n')
                        else:
                                contenido.append(linea)
                
        with open('files\\particles.txt', 'w') as archivo:
                archivo.writelines(contenido)

        print("partícula modificada")
        root.destroy()


def modifyParticle2(nombre):

    root = Tk()
    root.title("Modify Particle")
    root.config(width="700", height="500",background = "#38b382")
    root.iconbitmap('images\\icon.ico')
    root.resizable(0,0)
    
    frame1 = Frame(root)
    frame1.place(x = 200, y = 175)

    labelnombre = Label(frame1,text = "Nombre de la partícula: ")
    labelnombre.grid(row = 0,column = 0)

    EntryNombre = Entry(frame1)
    EntryNombre.grid(row= 0,column = 1,pady = 5,padx = 5)

    MinBtm = Label(frame1,text = "Btm mínimo de la partícula: ")
    MinBtm.grid(row = 1,column = 0)

    EntryMinBtm = Entry(frame1)
    EntryMinBtm.grid(row= 1,column = 1,pady = 5,padx = 5)

    MaxBtm = Label(frame1,text = "Btm máximo de la partícula: ")
    MaxBtm.grid(row = 2,column = 0)

    EntryMaxBtm = Entry(frame1)
    EntryMaxBtm.grid(row= 2,column = 1, pady = 5,padx = 5)

    margin = Label(frame1,text = "Margen de borde: ")
    margin.grid(row = 3,column = 0)

    EntryMargin = Entry(frame1)
    EntryMargin.grid(row= 3,column = 1, pady = 5,padx = 5)

    modifyFile_arg = partial(modifyFile,nombre,EntryNombre,EntryMinBtm,EntryMaxBtm,EntryMargin,root)
    button = Button(frame1,text = "send", width = 15,command = modifyFile_arg)
    button.grid(row = 4, column = 0)

def modifyParticle1():

    root = Tk()
    root.title("Modify Particle")
    root.config(width="700", height="500",background = "#38b382")
    root.iconbitmap('images\\icon.ico')
    root.resizable(0,0)
    
    frame1 = Frame(root)
    frame1.place(x = 200, y = 175)

    nombre = Label(frame1,text = "Nombre de la partícula a modificar: ")
    nombre.grid(row = 0,column = 0)

    EntryNombre = Entry(frame1)
    EntryNombre.grid(row= 0,column = 1,pady = 5,padx = 5)


    modifyParticle_arg = partial(modifyParticle2,EntryNombre)
    button = Button(frame1,text = "send", width = 15,command = modifyParticle_arg)
    button.grid(row = 1, column = 0)



#--------------------------------------Setting up interface------------------------------------#

root = Tk()
root.title("Menú principal")
root.config(width="700", height="500",background = "#38b382")
root.iconbitmap('images\\icon.ico')
root.resizable(0,0)

frame1 = Frame(root)
frame1.place(x = 150, y = 100)

foto = PhotoImage(file="images\\icono.png")
Label(frame1,image=foto).grid(row = 0, column = 0)



ParticleButton = Button(frame1, text = 'Identificar un tipo de partícula', width = 62,command = particle)
ParticleButton.grid(row=1,column=0)

addButton = Button(frame1,text ='Añadir un tipo de material',width = 62,command = addParticle)
addButton.grid(row = 2, column = 0)

addButton = Button(frame1,text ='Modificar un tipo de material',width = 62,command = modifyParticle1)
addButton.grid(row = 3, column = 0)

root.mainloop()
