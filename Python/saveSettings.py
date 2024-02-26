# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 19:47:38 2024

@author: bahman
"""

import tkinter as tk
from tkinter import filedialog


def browse_file(entry):
    filename = filedialog.askopenfilename()
    entry.delete(0, tk.END)
    entry.insert(0, filename)


def browse_directory(entry):
    directory = filedialog.askdirectory()
    entry.delete(0, tk.END)
    entry.insert(0, directory)

#def toggle_state0(*args):
#    if var.get() == 1:
#        input_b_entry.config(state='normal')
#        browse_button.config(state='normal')
#    else:
#        input_b_entry.config(state='disabled')
#        browse_button.config(state='disabled')
        
def toggle_state(input_b_entry, browse_button, var):
    print("in toggle_state")
    if var.get() == 1:
        input_b_entry.config(state='normal')
        browse_button.config(state='normal')
    else:
        input_b_entry.config(state='disabled')
        browse_button.config(state='disabled')        

#def toggle_state_replica_exchange0(*args):
#    if num_replicas_var.get() > 1:
#        max_temp_entry.config(state='normal')
#        exchange_attempts_entry.config(state='normal')
#    else:
#        max_temp_entry.config(state='disabled')
#        exchange_attempts_entry.config(state='disabled')

def toggle_state_replica_exchange(num_replicas_var, max_temp_entry, exchange_attempts_entry):
    if num_replicas_var.get() > 1:
        max_temp_entry.config(state='normal')
        exchange_attempts_entry.config(state='normal')
    else:
        max_temp_entry.config(state='disabled')
        exchange_attempts_entry.config(state='disabled')

def save_info(exchange_attempts_entry, number_of_iteration_entry, size_of_vector_entry, input_a_entry, input_b_entry, ising_or_qubo_var, num_replicas_entry, min_temp_entry, max_temp_entry, output_dir_entry):
    if int(exchange_attempts_entry.get()) >= int(number_of_iteration_entry.get()):
        print("Exchange attempts must be smaller than number of iterations.")
        return
    info = {
        "SizeOfVector": size_of_vector_entry.get(),
        "InputA": input_a_entry.get(),
        "InputB": input_b_entry.get(),
        "IsingorQUBO": ising_or_qubo_var.get(),
        "num_replicas": num_replicas_entry.get(),
        "minTemp": min_temp_entry.get(),
        "maxTemp": max_temp_entry.get(),
        "exchange_attempts": exchange_attempts_entry.get(),
        "NumberOfIteration": number_of_iteration_entry.get(),
        "outputDir": output_dir_entry.get()
    }
    with open("settings666.txt", "w") as f:
        for key, value in info.items():
            f.write(f"{key}: {value}\n")




def show_new_settings (root):
    new_window = tk.Toplevel(root)
    new_window.title("New Window")
    
    #label = tk.Label(new_window, text="This is a new window!")
    #label.pack()
    
    new_window.geometry("600x300")


    tk.Label(new_window, text="Size Of Vector").grid(row=1, column=0)
    size_of_vector_entry = tk.Entry(new_window)
    size_of_vector_entry.grid(row=1, column=1)
    
    
    
    tk.Label(new_window, text="Path to Matrix A").grid(row=2, column=0)
    input_a_entry = tk.Entry(new_window)
    input_a_entry.grid(row=2, column=1)

    #browse_button = tk.Button(new_window, text="Browse", command=browse_file)
    browse_button = tk.Button(new_window, text="Browse", command=lambda: browse_file(input_a_entry))
    browse_button.grid(row=2, column=2)

    


    tk.Label(new_window, text="Path to bias file").grid(row=3, column=1)
    input_b_entry = tk.Entry(new_window)
    input_b_entry.grid(row=3, column=2)

    browse_button = tk.Button(new_window, text="Browse", command=lambda: browse_file(input_b_entry))
    #browse_button = tk.Button(new_window, text="Browse", command=browse_file)
    browse_button.grid(row=3, column=3)
    

    var = tk.IntVar(value=1)
    var.trace('w', toggle_state(input_b_entry, browse_button, var))    
    cb = tk.Checkbutton(new_window, text="Bias?", variable=var, onvalue=1, offvalue=0)
    cb.grid(row=3, column=0)

    tk.Label(new_window, text="Number Of Iteration").grid(row=4, column=0)
    number_of_iteration_entry = tk.Entry(new_window)
    number_of_iteration_entry.grid(row=4, column=1)

    tk.Label(new_window, text="Execute Mode").grid(row=5, column=0)
    ising_or_qubo_var = tk.StringVar(new_window)
    ising_or_qubo_var.set("QUBOGPU")  # default value
    ising_or_qubo_option = tk.OptionMenu(new_window, ising_or_qubo_var, "QUBOGPU", "ISING", "QUBO")
    ising_or_qubo_option.grid(row=5, column=1)


    tk.Label(new_window, text="Minimum Temperature").grid(row=7, column=0)
    min_temp_entry = tk.Entry(new_window)
    min_temp_entry.grid(row=7, column=1)

    tk.Label(new_window, text="Maximum Temperature").grid(row=8, column=0)
    max_temp_entry = tk.Entry(new_window, state='disabled')
    max_temp_entry.grid(row=8, column=1)

    tk.Label(new_window, text="Exchange attempts").grid(row=9, column=0)
    exchange_attempts_entry = tk.Entry(new_window, state='disabled')
    exchange_attempts_entry.grid(row=9, column=1)



    tk.Label(new_window, text="Number of replicas").grid(row=6, column=0)
    num_replicas_var = tk.IntVar()
    num_replicas_var.trace('w', toggle_state_replica_exchange(num_replicas_var, max_temp_entry, exchange_attempts_entry))
    num_replicas_entry = tk.Entry(new_window, textvariable=num_replicas_var)
    num_replicas_entry.grid(row=6, column=1)




    tk.Label(new_window, text="Output folder").grid(row=10, column=0)
    output_dir_entry = tk.Entry(new_window)
    output_dir_entry.grid(row=10, column=1)

    browse_button_3 = tk.Button(new_window, text="Browse", command=lambda: browse_directory(output_dir_entry))
    browse_button_3.grid(row=10, column=2)

    #save_button = tk.Button(new_window, text="Save Info", command=save_info(exchange_attempts_entry, number_of_iteration_entry, size_of_vector_entry, input_a_entry, input_b_entry, ising_or_qubo_var, num_replicas_entry, min_temp_entry, max_temp_entry, output_dir_entry))
    #save_button.grid(row=11, column=1)
    

