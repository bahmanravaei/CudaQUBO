# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 20:38:40 2024

@author: bahman
"""


import subprocess

import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from tkinter import messagebox
from tkinter import PhotoImage
import os
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd


def toggle_visibility(column_index, figure):
    # Toggle visibility of the specified column in the plot
    plot.lines[column_index].set_visible(not plot.lines[column_index].get_visible())
    figure.canvas.draw_idle()

def draw_chart(output_dir, window):
    chart_window = tk.Toplevel(window)

    # Generate or load data for the chart
    file_path =output_dir + '\Energy.csv'  # Replace with the actual path to your CSV file
    df = pd.read_csv(file_path, header=None)

    # Create a Matplotlib figure
    figure = Figure(figsize=(8, 6), dpi=100)
    global plot  # Global variable to access the plot from other functions
    plot = figure.add_subplot(1, 1, 1)

    # Plot each column and create a checkbox for each column
    checkboxes = []
    for i, column in enumerate(df.columns):
        plot.plot(df.index, df[column], label=column)

        # Create a checkbox for each column
        var = tk.IntVar()
        checkbox = tk.Checkbutton(chart_window, text=column, variable=var, command=lambda i=i: toggle_visibility(i, figure))
        checkbox.select()  # By default, all columns are visible
        checkbox.pack(anchor=tk.W, padx=5)
        #checkbox.pack(side=tk.LEFT, anchor=tk.W, pady=5)
        checkboxes.append((var, checkbox))

    plot.set_title("Chart Title")
    plot.set_xlabel("X-axis Label")
    plot.set_ylabel("Y-axis Label")
    plot.legend()

    # Embed the Matplotlib figure in the Tkinter window
    canvas = FigureCanvasTkAgg(figure, master=chart_window)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

def relative_path_(user_path):
    current_directory = os.getcwd()
    user_path = os.path.abspath(user_path)  # Convert to absolute path for accurate comparison
    
    # Check if the user path is a subdirectory of the current directory
    if user_path.startswith(current_directory):
        #print(f"{user_path} is a subdirectory of {current_directory}")
        relative_path = os.path.relpath(user_path, current_directory)
        return relative_path
    else:
        #print(f"{user_path} is not a subdirectory of {current_directory}")
        return user_path


def browse_file(entry):
    filename = filedialog.askopenfilename()
    filename= relative_path_(filename)
        
    entry.delete(0, tk.END)
    entry.insert(0, filename)


def browse_directory(entry):
    directory = filedialog.askdirectory()
    directory= relative_path_(directory)
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
        input_b_entry.config(state=tk.NORMAL)
        browse_button.config(state=tk.NORMAL)
    else:
        input_b_entry.delete(0,tk.END)
        input_b_entry.config(state=tk.DISABLED)
        browse_button.config(state=tk.DISABLED) 
        

#def toggle_state_replica_exchange0(*args):
#    if num_replicas_var.get() > 1:
#        max_temp_entry.config(state='normal')
#        exchange_attempts_entry.config(state='normal')
#    else:
#        max_temp_entry.config(state='disabled')
#        exchange_attempts_entry.config(state='disabled')

def toggle_state_replica_exchange(num_replicas_var, max_temp_entry, exchange_attempts_entry):
    print("in toggle_state_replica_exchange")
    if num_replicas_var.get() > 1:
        max_temp_entry.config(state='normal')
        exchange_attempts_entry.config(state='normal')
    else:
        max_temp_entry.delete(0, tk.END)
        exchange_attempts_entry.delete(0, tk.END)
        max_temp_entry.config(state='disabled')
        exchange_attempts_entry.config(state='disabled')


def read_info(file_path="settings.txt"):
    info = {}

    try:
        with open(file_path, "r") as f:
            for line in f:
                key, value = line.strip().split(": ")
                info[key] = value
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error reading file: {e}")

    return info


def assign_value(entry, value_to_assign):
    entry.config(state=tk.NORMAL)
    entry.delete(0, tk.END)
    entry.insert(0, value_to_assign)


def save_info(exchange_attempts_entry, number_of_iteration_entry, size_of_vector_entry, input_a_entry, input_b_entry, ising_or_qubo_var, num_replicas_entry, min_temp_entry, max_temp_entry, output_dir_entry):
    print("In save_info")
    
    
    
    infoSave=[["SizeOfVector", size_of_vector_entry],
              ["InputA", input_a_entry],
              ["InputB", input_b_entry],
              ["IsingorQUBO", ising_or_qubo_var],
              ["num_replicas", num_replicas_entry],
              ["minTemp", min_temp_entry],
              ["maxTemp", max_temp_entry],
              ["exchange_attempts", exchange_attempts_entry],
              ["NumberOfIteration", number_of_iteration_entry],
              ["outputDir", output_dir_entry]]
    
    neccesary_item=[0, 1, 4, 5, 8, 9]
    info = {}
    for i in range(len(infoSave)):
        item_value= infoSave[i][1].get()
        if i in neccesary_item:
            if item_value=="":
                messagebox.showinfo("Alert", "No" +infoSave[i][0])
                return
        
        if item_value:
            info[infoSave[i][0]] = infoSave[i][1].get()
        else:
            print("No"+infoSave[i][0])
    
    print(info)
    

    if exchange_attempts_entry.get()!="" and int(exchange_attempts_entry.get()) >= int(number_of_iteration_entry.get()):
        messagebox.showinfo("Alert","Exchange attempts must be smaller than number of iterations.")
        return
    
    
    with open("settings.txt", "w") as f:
        for key, value in info.items():
            f.write(f"{key}: {value}\n")
            
    return
#    info = {
#        "SizeOfVector": size_of_vector_entry.get(),
#        "InputA": input_a_entry.get(),
#        "InputB": input_b_entry.get(),
#        "IsingorQUBO": ising_or_qubo_var.get(),
#        "num_replicas": num_replicas_entry.get(),
#        "minTemp": min_temp_entry.get(),
#        "maxTemp": max_temp_entry.get(),
#       "exchange_attempts": exchange_attempts_entry.get(),
#        "NumberOfIteration": number_of_iteration_entry.get(),
#        "outputDir": output_dir_entry.get()
#    }
    

def validate_int(value):
    try:
        if value == "" or int(value) >= 0:
            return True
        else:
            return False
    except ValueError:
        return False


def show_new_settings (edit_flag=False):
    new_window = tk.Toplevel(root)
    #new_window=tk.Tk()
    
    
    #label = tk.Label(new_window, text="This is a new window!")
    #label.pack()
    
    new_window.geometry("600x300")
    set_background_image(new_window)

    tk.Label(new_window, text="Size Of Vector").grid(row=1, column=0)
    size_of_vector_entry = tk.Entry(new_window)
    size_of_vector_entry.grid(row=1, column=1)
    
    
    
    tk.Label(new_window, text="Path to Matrix A").grid(row=2, column=0)
    input_a_entry = tk.Entry(new_window)
    input_a_entry.grid(row=2, column=1)

    #browse_button = tk.Button(new_window, text="Browse", command=browse_file)
    browse_button = tk.Button(new_window, text="Browse", command=lambda: browse_file(input_a_entry))
    browse_button.grid(row=2, column=2)

    

    #print("Before toggle")
    checkbox_var = tk.IntVar()
    #var.trace('w', toggle_state(input_b_entry, browse_button, var))    
    cb = tk.Checkbutton(new_window, text="Bias?", variable=checkbox_var, command=lambda: toggle_state(input_b_entry, browse_button, checkbox_var))
    cb.grid(row=3, column=0)


    tk.Label(new_window, text="Path to bias file").grid(row=3, column=1)
    input_b_entry = tk.Entry(new_window, state= tk.DISABLED)
    input_b_entry.grid(row=3, column=2)

    browse_button = tk.Button(new_window, text="Browse", command=lambda: browse_file(input_b_entry))
    #browse_button = tk.Button(new_window, text="Browse", command=browse_file)
    browse_button.grid(row=3, column=3)
    
   

  

    tk.Label(new_window, text="Number Of Iteration").grid(row=4, column=0)
    number_of_iteration_entry = tk.Entry(new_window)
    number_of_iteration_entry.grid(row=4, column=1)

    tk.Label(new_window, text="Execute Mode").grid(row=5, column=0)
    ising_or_qubo_var = tk.StringVar(new_window)
    ising_or_qubo_var.set("QUBOGPU")  # default value
    ising_or_qubo_option = tk.OptionMenu(new_window, ising_or_qubo_var, "QUBOGPU", "ISING", "QUBO")
    ising_or_qubo_option.grid(row=5, column=1)

    
#checkbox_var = tk.IntVar()
#checkbox = tk.Checkbutton(new_window, text="Enable Entry", variable=checkbox_var, command=lambda: toggle_entry(entry, checkbox_var))
#entry = tk.Entry(new_window, state=tk.DISABLED)



    tk.Label(new_window, text="Number of replicas").grid(row=6, column=0)
    num_replicas_var = tk.IntVar()
    #num_replicas_var.trace('w', toggle_state_replica_exchange(num_replicas_var, max_temp_entry, exchange_attempts_entry))
    num_replicas_entry = tk.Entry(new_window, textvariable=num_replicas_var)#, command=lambda: toggle_state_replica_exchange(num_replicas_var, max_temp_entry, exchange_attempts_entry))
    num_replicas_entry.grid(row=6, column=1)

    tk.Label(new_window, text="Minimum Temperature").grid(row=7, column=0)
    min_temp_entry = tk.Entry(new_window)
    min_temp_entry.grid(row=7, column=1)

    tk.Label(new_window, text="Maximum Temperature").grid(row=8, column=0)
    max_temp_entry = tk.Entry(new_window, state='disabled')
    max_temp_entry.grid(row=8, column=1)

    tk.Label(new_window, text="Exchange attempts").grid(row=9, column=0)
    exchange_attempts_entry = tk.Entry(new_window, state='disabled')
    exchange_attempts_entry.grid(row=9, column=1)
    
    #entry1.bind('<KeyRelease>', lambda *args: toggle_entry2(entry1, entry2))
    num_replicas_entry.bind('<KeyRelease>', lambda *args: toggle_state_replica_exchange(num_replicas_var, max_temp_entry, exchange_attempts_entry))


    tk.Label(new_window, text="Output folder").grid(row=10, column=0)
    output_dir_entry = tk.Entry(new_window)
    output_dir_entry.grid(row=10, column=1)

    browse_button_3 = tk.Button(new_window, text="Browse", command=lambda: browse_directory(output_dir_entry))
    browse_button_3.grid(row=10, column=2)

    save_button = tk.Button(new_window, text="Save Info", command=lambda: save_info(exchange_attempts_entry, number_of_iteration_entry, size_of_vector_entry, input_a_entry, input_b_entry, ising_or_qubo_var, num_replicas_entry, min_temp_entry, max_temp_entry, output_dir_entry))
    save_button.grid(row=11, column=1)
    
    
    infoSave={"SizeOfVector": size_of_vector_entry,
              "InputA": input_a_entry,
              "InputB": input_b_entry,
              "IsingorQUBO": ising_or_qubo_var,
              "num_replicas": num_replicas_entry,
              "minTemp": min_temp_entry,
              "maxTemp": max_temp_entry,
              "exchange_attempts": exchange_attempts_entry,
              "NumberOfIteration": number_of_iteration_entry,
              "outputDir": output_dir_entry}
    
    if edit_flag == True:
        new_window.title("Edit Settings")
        saved_info=read_info()
        for key, value in saved_info.items():
            if isinstance(infoSave[key], tk.Entry):
                assign_value(infoSave[key], value)
            elif isinstance(infoSave[key], tk.StringVar):
                infoSave[key].set(value)
            
            if key == "InputB":
                checkbox_var.set(1)
                
                
    else:
        new_window.title("New Settings")
        
    
    #new_window.mainloop()


def show_table():
    # Create a new window
    table_window = tk.Toplevel(root)

    # Create a treeview widget
    treeview = ttk.Treeview(table_window, columns=("Value"), show="headings")
    treeview.heading("Value", text="Value")
    treeview.pack()

    # Read the data from the file
    with open("settings.txt", "r") as f:
        for line in f:
            key, value = line.strip().split(": ")
            # Insert the data into the treeview
            treeview.insert("", "end", text=key, values=(value,))



def run_qubo(analyze_button):
    print ("run QUBO")
    exe_path = 'CudaQUBO.exe'
    try:
        subprocess.run(exe_path, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
    except FileNotFoundError:
        print(f"Error: The specified .exe file '{exe_path}' was not found.")
    messagebox.showinfo("Alert","the result for QUBO is ready")
    analyze_button.config(state=tk.NORMAL)
    return



def get_output_dir(file_path='settings.txt'):
    try:
        with open(file_path, 'r') as file:
            for line in file:
                key, value = map(str.strip, line.split(':', 1))
                if key == 'outputDir':
                    return value
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except Exception as e:
        print(f"Error while reading '{file_path}': {e}")

    return None

def analyze():
    output_dir = get_output_dir()
    draw_chart(output_dir, root)
    return

def main():
    print("Hello, World!")
    
    
    
    #show_table_button = tk.Button(root, text="Show Table", command=show_table(root))
    #show_table_button.grid(row=1, column=1)
    
    #save_button = tk.Button(root, text="Edit Setting", command=save_info)
    #save_button.grid(row=11, column=1)
    
    new_setting_button = tk.Button(root, text="New Settings", command=show_new_settings)
    new_setting_button.grid(row=1, column=1)
    
    edit_setting_button = tk.Button(root, text="Edit Settings", command=lambda: show_new_settings(True))
    edit_setting_button.grid(row=2, column=2)
    
    execute_setting_button = tk.Button(root, text="Execute QUBO", command=lambda: run_qubo(analyze_button))
    execute_setting_button.grid(row=3, column=3)
    
    #analyze_button = tk.Button(root, text="Analyze the results", command=analyze, state='disabled')
    analyze_button = tk.Button(root, text="Analyze the results", command=analyze)
    analyze_button.grid(row=4, column=4)


def set_background_image(window):
    # Replace 'background_image.png' with the path to your image file
    image_path = 'Ising-tartan.png'
    
    # Create a PhotoImage object from the image file
    background_image = PhotoImage(file=image_path)

    # Set the image as the window background
    background_label = tk.Label(window, image=background_image)
    background_label.place(relwidth=1, relheight=1)

    # Ensure the image is not garbage collected
    window.image = background_image

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Main Window")
    root.geometry("600x300")
    #set_background_image(root)
    
    main()
    
    root.mainloop()