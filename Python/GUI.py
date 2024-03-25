# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 20:38:40 2024

@author: bahman
"""


import subprocess
import my_utility_function as my_f

import tkinter as tk
#from tkinter import filedialog
from tkinter import ttk
from tkinter import messagebox
#from tkinter import PhotoImage
import os
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
#from PIL import Image, ImageTk
import MaxCutToIsing as maxCut
import constant as ct


debug_list = ["DEBUG_RANDOM_FLIP",
            "DEBUG_SELECTED_FLIP",
            "DEBUG_FIND_BEST_ENERGY",
            "DEBUG_UPDATE_H",
            "DEBUG_EXCHANGE",
            "DEBUG_SAVE_DEVICE_RESULT",
            "DEBUG_DELTA_FLIP",
            "DEBUG_INIT_CONFIG",
            "DEBUG_SAVE_W_MATRIX",
            "DEBUG_WARNNING_AND_INFO",
            "DEBUG_FINAL_CONFIG",
            "DEBUG_ENERGY_RECORD_LOG",
            "DEBUG_MAGNET_RECORD_LOG"]


neccesary_item=["SizeOfVector", "InputA", "num_replicas", "minTemp", "NumberOfIteration", "outputDir"]

def toggle_visibility(column_index, figure):
    # Toggle visibility of the specified column in the plot
    plot.lines[column_index].set_visible(not plot.lines[column_index].get_visible())
    figure.canvas.draw_idle()

def draw_chart(output_dir, window):
    chart_window = tk.Toplevel(window)
    chart_window.title("Charts")
    # Top Frame
    top_frame = tk.Frame(chart_window)
    top_frame.grid(row=0, column=0, pady=10)  # Adjust pady as needed
   
    # Bottom Frame
    bottom_frame = tk.Frame(chart_window)
    bottom_frame.grid(row=1, column=0, pady=10)  # Adjust pady as needed


    # Generate or load data for the chart
    file_path =output_dir + '\Energy.csv'  # Replace with the actual path to your CSV file
    df = pd.read_csv(file_path, header=None)

    # Create a Matplotlib figure
    figure = Figure(figsize=(8, 6), dpi=100)
    global plot  # Global variable to access the plot from other functions
    plot = figure.add_subplot(1, 1, 1)

    # Plot each column and create a checkbox for each column
    checkboxes = []
    for i, col in enumerate(df.columns):
        plot.plot(df.index, df[col], label="replica "+str(col))

        # Create a checkbox for each column
        var = tk.IntVar()
        #checkbox = tk.Checkbutton(chart_window, text=column, variable=var, command=lambda i=i: toggle_visibility(i, figure))
        checkbox = tk.Checkbutton(top_frame, text="replica "+str(col), variable=var, command=lambda i=i: toggle_visibility(i, figure))
        checkbox.select()  # By default, all columns are visible
        #checkbox.pack(anchor=tk.W, padx=5)
        #checkbox.pack(side=tk.LEFT, anchor=tk.W, pady=5)
        checkboxes.append((var, checkbox))
        print(col)
        checkbox.grid(row=int(i/8), column=i%8)

    plot.set_title(" ")
    plot.set_xlabel("Number of Iterations")
    plot.set_ylabel("Energy")
    plot.legend()

    # Embed the Matplotlib figure in the Tkinter window
    #canvas = FigureCanvasTkAgg(figure, master=chart_window)
    canvas = FigureCanvasTkAgg(figure, master=bottom_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)








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

def convert_key_for_debug(key, value):
    for i in range(len(debug_list)):
        item = debug_list[i]
        if item == value:
            index = i
            break
    
    key = "debugMode"+str(index)
    return key, 1

def read_infoV2(file_path="settings.txt"):
    info = {}
    flag_find_problem_type = False
    try:
        with open(file_path, "r") as f:
            for line in f:
                key, value = line.strip().split(": ")
                if key == "problemType":
                    flag_find_problem_type = True
                    if value == 'NONSYMETRIC':
                        value = 0
                if key == "debugMode":
                    key, value = convert_key_for_debug(key, value)
                info[key]=value
            if (flag_find_problem_type == False):
                info["problemType"] = 1
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error reading file: {e}")
        
    print("in read_infoV2", info)

    return info


def is_settings_file_exists(file_path):
    #file_path = "settings.txt"
    return os.path.exists(file_path)

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

def convert_it_to_list(infoSave):
    info_as_list = []
    

    

    
    for key, value in infoSave.items():
        if(key[0:9]=="debugMode"):
            if (value.get()==1):
                index=int(key[9:])
                info_as_list.append(["debugMode", debug_list[index]])
        elif(key == 'problemType'):
            if (value.get()!=1):
                info_as_list.append(["problemType", "NONSYMETRIC"])
        else:
            info_as_list.append([key, value.get()])
        #if(key in get_item):
            
        #else:
         #   info_as_list.append([key, value])


    #print(info_as_list)
    return info_as_list


def validate_items(infoSave):
    
    Flag = True
    number_of_iteration_value = 0
    exchange_attempts_value = 0
    for i in range(len(infoSave)):
        item_key = infoSave[i][0]
        item_value = infoSave[i][1]
        if item_key in neccesary_item:
            if item_value=="":
                messagebox.showinfo("Alert", "No  " +item_key)
                Flag = False
        
        if (item_key == "exchange_attempts"  and item_value !=""):
            exchange_attempts_value = int(item_value)
            
        if (item_key == "NumberOfIteration"  and item_value !=""):
            number_of_iteration_value = int(item_value)
                
    if (exchange_attempts_value >= number_of_iteration_value):
            messagebox.showinfo("Alert","Exchange attempts must be smaller than number of iterations.")
            Flag = False
        
    return Flag
    
def remove_item_without_data(infoSave):
    new_info = []
    for item in infoSave:
        if item[1] != "":
            new_info.append(item)
    return new_info

def save_info2(infoSave):
    
    infoSave = convert_it_to_list(infoSave)
    print(infoSave)
    flag=validate_items(infoSave)
    if (flag==False):
        return
    infoSave = remove_item_without_data(infoSave)
    print(infoSave)
    with open("settings.txt", "w") as f:
        for item in infoSave:
            key = item[0]
            value = item[1]
            f.write(f"{key}: {value}\n")
    return

        

def validate_int(value):
    try:
        if value == "" or int(value) >= 0:
            return True
        else:
            return False
    except ValueError:
        return False

def close_new_window(window):
    # Unlock the parent window
    root.attributes("-disabled", False)
    window.destroy()

def show_new_settings (cuda_info, edit_flag=False):
    new_window = tk.Toplevel(root)
    
    root.attributes("-disabled", True)
    new_window.protocol("WM_DELETE_WINDOW", lambda: close_new_window(new_window))
    
    new_window.geometry("550x700")
    my_f.set_background_image(new_window)
    
    #config_frame = tk.Frame(root, bd=2, relief=tk.GROOVE, highlightthickness=2, highlightbackground="black")
    #config_frame.grid(row=row_position, column=0, pady=10)  # Adjust pady as needed
    
    frame_config = tk.Frame(new_window, bd=2, relief=tk.GROOVE, highlightthickness=2, highlightbackground="black")
    frame_config.grid(row=0, column=0, pady=10, padx=10)  # Adjust pady as needed
    
    label_frame_config = tk.Label(frame_config, text="Enter configuration for QUBO problem", font=("Helvetica", 12, "bold"))
    label_frame_config.grid(row=0, column=0, columnspan=5, sticky='w', padx=(10, 0), pady=(0, 10))
    
    #new_window=tk.Tk()
    
    
    #label = tk.Label(new_window, text="This is a new window!")
    #label.pack()
    


    tk.Label(frame_config, text="Size Of Vector").grid(row=1, column=0, sticky="w")
    size_of_vector_entry = tk.Entry(frame_config)
    size_of_vector_entry.grid(row=1, column=1)
    
    tk.Label(frame_config, text="Maximum: " + str(cuda_info['max_threads_per_block'])).grid(row=1, column=2)
    
    
    tk.Label(frame_config, text="Path to Matrix A").grid(row=2, column=0, sticky="w")
    input_a_entry = tk.Entry(frame_config)
    input_a_entry.grid(row=2, column=1)

    #browse_button = tk.Button(new_window, text="Browse", command=browse_file)
    browse_button = tk.Button(frame_config, text="Browse", command=lambda: my_f.browse_file(input_a_entry))
    browse_button.grid(row=2, column=2)


    checkbox_symmetric_var = tk.IntVar()
    cb_symmetric = tk.Checkbutton(frame_config, text="Symmetric?", variable=checkbox_symmetric_var)
    cb_symmetric.grid(row=2, column=3, sticky="w")    


    #print("Before toggle")
    checkbox_var = tk.IntVar()
    #var.trace('w', toggle_state(input_b_entry, browse_button, var))    
    cb = tk.Checkbutton(frame_config, text="Bias?", variable=checkbox_var, command=lambda: toggle_state(input_b_entry, browse_button, checkbox_var))
    cb.grid(row=3, column=0, sticky="w")


    tk.Label(frame_config, text="Path to bias file").grid(row=3, column=1, sticky="w")
    input_b_entry = tk.Entry(frame_config, state= tk.DISABLED)
    input_b_entry.grid(row=3, column=2)

    browse_button = tk.Button(frame_config, text="Browse", command=lambda: my_f.browse_file(input_b_entry))
    #browse_button = tk.Button(new_window, text="Browse", command=browse_file)
    browse_button.grid(row=3, column=3)
    
   

  

    tk.Label(frame_config, text="Number Of Iteration").grid(row=4, column=0, sticky="w")
    number_of_iteration_entry = tk.Entry(frame_config)
    number_of_iteration_entry.grid(row=4, column=1)

    tk.Label(frame_config, text="Execute Mode").grid(row=5, column=0, sticky="w")
    ising_or_qubo_var = tk.StringVar(frame_config)
    ising_or_qubo_var.set("QUBOGPUFULL")  # default value
    ising_or_qubo_option = tk.OptionMenu(frame_config, ising_or_qubo_var, "QUBOGPUFULL" , "QUBOGPU", "ISING", "QUBO")
    ising_or_qubo_option.grid(row=5, column=1)

    
#checkbox_var = tk.IntVar()
#checkbox = tk.Checkbutton(new_window, text="Enable Entry", variable=checkbox_var, command=lambda: toggle_entry(entry, checkbox_var))
#entry = tk.Entry(new_window, state=tk.DISABLED)



    tk.Label(frame_config, text="Number of replicas").grid(row=6, column=0, sticky="w")
    num_replicas_var = tk.IntVar()
    #num_replicas_var.trace('w', toggle_state_replica_exchange(num_replicas_var, max_temp_entry, exchange_attempts_entry))
    num_replicas_entry = tk.Entry(frame_config, textvariable=num_replicas_var)#, command=lambda: toggle_state_replica_exchange(num_replicas_var, max_temp_entry, exchange_attempts_entry))
    num_replicas_entry.grid(row=6, column=1)

    tk.Label(frame_config, text="Minimum Temperature").grid(row=7, column=0, sticky="w")
    min_temp_entry = tk.Entry(frame_config)
    min_temp_entry.grid(row=7, column=1)

    tk.Label(frame_config, text="Maximum Temperature").grid(row=8, column=0, sticky="w")
    max_temp_entry = tk.Entry(frame_config, state='disabled')
    max_temp_entry.grid(row=8, column=1)

    tk.Label(frame_config, text="Exchange attempts").grid(row=9, column=0, sticky="w")
    exchange_attempts_entry = tk.Entry(frame_config, state='disabled')
    exchange_attempts_entry.grid(row=9, column=1)
    
    #entry1.bind('<KeyRelease>', lambda *args: toggle_entry2(entry1, entry2))
    num_replicas_entry.bind('<KeyRelease>', lambda *args: toggle_state_replica_exchange(num_replicas_var, max_temp_entry, exchange_attempts_entry))


    tk.Label(frame_config, text="Output folder").grid(row=10, column=0, sticky="w")
    output_dir_entry = tk.Entry(frame_config)
    output_dir_entry.grid(row=10, column=1)

    browse_button_3 = tk.Button(frame_config, text="Browse", command=lambda: my_f.browse_directory(output_dir_entry))
    browse_button_3.grid(row=10, column=2)


    
    ##########################################################################
    frame_debug = tk.Frame(new_window, bd=2, relief=tk.GROOVE, highlightthickness=2, highlightbackground="black")
    frame_debug.grid(row=1, column=0, pady=10, padx=10)  # Adjust pady as needed
    
    label_frame_debug = tk.Label(frame_debug, text="Selecting items from the following lists to record logs", font=("Helvetica", 12, "bold"))
    label_frame_debug.grid(row=0, column=0, columnspan=5, sticky='w', padx=(10, 0), pady=(0, 10))
    
    ##########################################################################
    frame_debug_code = tk.Frame(frame_debug, bd=2, relief=tk.GROOVE, highlightthickness=2, highlightbackground="black")
    frame_debug_code.grid(row=1, column=0, pady=10)  # Adjust pady as needed
    
    label_frame_debug = tk.Label(frame_debug_code, text="Select if your want to \n debug the Cuda code", font=("Helvetica", 12, "bold"))
    label_frame_debug.grid(row=0, column=0, columnspan=5, sticky='w', padx=(10, 0), pady=(0, 10))
    
    
    checkbox_log_random_flip_var = tk.IntVar()
    cb_log_random_flip = tk.Checkbutton(frame_debug_code, text="Log random flip?", variable=checkbox_log_random_flip_var)
    cb_log_random_flip.grid(row=1, column=0, sticky="w") 
    
    
    checkbox_log_delta_flip_var = tk.IntVar()
    cb_log_delta_flip = tk.Checkbutton(frame_debug_code, text="Log less energy flip?", variable=checkbox_log_delta_flip_var)
    cb_log_delta_flip.grid(row=2, column=0, sticky="w")
    
    checkbox_log_selected_flip_var = tk.IntVar()
    cb_log_selected_flip = tk.Checkbutton(frame_debug_code, text="Log selected bit flip?", variable=checkbox_log_selected_flip_var)
    cb_log_selected_flip.grid(row=3, column=0, sticky="w")
    
    checkbox_log_local_optima_energy_var = tk.IntVar()
    cb_log_local_optima_energy = tk.Checkbutton(frame_debug_code, text="Log local optima energy?", variable=checkbox_log_local_optima_energy_var)
    cb_log_local_optima_energy.grid(row=4, column=0, sticky="w")
    
    checkbox_log_update_h_var = tk.IntVar()
    cb_log_update_h = tk.Checkbutton(frame_debug_code, text="Log update H?", variable=checkbox_log_update_h_var)
    cb_log_update_h.grid(row=5, column=0, sticky="w")
    
    checkbox_log_exchange_temperature_var = tk.IntVar()
    cb_log_exchange_temperature = tk.Checkbutton(frame_debug_code, text="Log exchange temperature?", variable=checkbox_log_exchange_temperature_var)
    cb_log_exchange_temperature.grid(row=6, column=0, sticky="w")
    
    checkbox_log_gpu_final_result_var = tk.IntVar()
    cb_log_gpu_final_result = tk.Checkbutton(frame_debug_code, text="Log GPU final results?", variable=checkbox_log_gpu_final_result_var)
    cb_log_gpu_final_result.grid(row=7, column=0, sticky="w")

    ###########################################################################
    frame_debug_warning_info = tk.Frame(frame_debug, bd=2, relief=tk.GROOVE, highlightthickness=2, highlightbackground="black")
    frame_debug_warning_info.grid(row=1, column=1, pady=10, padx=20)  # Adjust pady as needed
    
    label_frame_debug_warning_info = tk.Label(frame_debug_warning_info, text="Select to record logs", font=("Helvetica", 12, "bold"))
    label_frame_debug_warning_info.grid(row=0, column=0, columnspan=5, sticky='w', padx=(10, 0), pady=(0, 10))
    
    checkbox_log_init_config_var = tk.IntVar()
    cb_log_init_config = tk.Checkbutton(frame_debug_warning_info, text="Log initial configuration?", variable=checkbox_log_init_config_var)
    cb_log_init_config.grid(row=1, column=0, sticky="w")    
    
    checkbox_log_final_config_var = tk.IntVar()
    cb_log_final_config = tk.Checkbutton(frame_debug_warning_info, text="Log final config?", variable=checkbox_log_final_config_var)
    cb_log_final_config.grid(row=2, column=0, sticky="w") 
    
    checkbox_log_energy_record_var = tk.IntVar()
    cb_log_energy_record = tk.Checkbutton(frame_debug_warning_info, text="Log energy?", variable=checkbox_log_energy_record_var)
    cb_log_energy_record.grid(row=3, column=0, sticky="w")
    
    checkbox_log_magnet_record_var = tk.IntVar()
    cb_log_magnet_record = tk.Checkbutton(frame_debug_warning_info, text="Log magnetization?", variable=checkbox_log_magnet_record_var)
    cb_log_magnet_record.grid(row=4, column=0, sticky="w")
    
    
    configuration_dict={"SizeOfVector": size_of_vector_entry,
              "InputA": input_a_entry,
              "InputB": input_b_entry,
              "IsingorQUBO": ising_or_qubo_var,
              "num_replicas": num_replicas_entry,
              "minTemp": min_temp_entry,
              "maxTemp": max_temp_entry,
              "exchange_attempts": exchange_attempts_entry,
              "NumberOfIteration": number_of_iteration_entry,
              "outputDir": output_dir_entry,
              "problemType": checkbox_symmetric_var, 
              "debugMode0": checkbox_log_random_flip_var,
              "debugMode1": checkbox_log_selected_flip_var,
              "debugMode2": checkbox_log_local_optima_energy_var,
              "debugMode3": checkbox_log_update_h_var,
           	  "debugMode4": checkbox_log_exchange_temperature_var,
           	  "debugMode5": checkbox_log_gpu_final_result_var,
           	  "debugMode6": checkbox_log_delta_flip_var,
           	  "debugMode7": checkbox_log_init_config_var,
           	  "debugMode10": checkbox_log_final_config_var,
           	  "debugMode11": checkbox_log_energy_record_var,
           	  "debugMode12": checkbox_log_magnet_record_var}
    
    if edit_flag == True:
        new_window.title("Edit Settings")
        saved_info=read_infoV2()
        print(saved_info)
        for key, value in saved_info.items():
            if isinstance(configuration_dict[key], tk.Entry):
                assign_value(configuration_dict[key], value)
            elif isinstance(configuration_dict[key], tk.StringVar) or isinstance(configuration_dict[key], tk.IntVar):
                configuration_dict[key].set(value)
            
            if key == "InputB":
                checkbox_var.set(1)
                
                
    else:
        new_window.title("New Settings")
        
    
    #new_window.mainloop()
    #save_button = tk.Button(new_window, text="Save Info", command=lambda: save_info(exchange_attempts_entry, number_of_iteration_entry, size_of_vector_entry, input_a_entry, input_b_entry, ising_or_qubo_var, num_replicas_entry, min_temp_entry, max_temp_entry, output_dir_entry))
    save_button = tk.Button(new_window, text="Save Info", command=lambda: save_info2(configuration_dict))
    save_button.grid(row=11, column=0)

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


def convert_to_string(a_dictionary):
    print(a_dictionary)
    string = ""
    for key, value in a_dictionary.items():
        string += key + ": " + str(value) + "\n"
    return string

def run_qubo(analyze_button):
    if is_settings_file_exists("settings.txt")==False:
        print("Can not find the seetings.txt file !")
        messagebox.showinfo("Alert","Can not find the seetings.txt file !")
        return
    
    info_settings = read_info(file_path="settings.txt")
    
    setting_in_string_format = convert_to_string(info_settings)
    messagebox.showinfo("Alert", setting_in_string_format)
    
    
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




def convert_Frame(root):
    # Define some functions to covert different Graph Problem in QUBO or Ising
    convert_frame = tk.Frame(root, bd=2, relief=tk.GROOVE, highlightthickness=2, highlightbackground="black")
    convert_frame.grid(row=0, column=0, pady=10)  # Adjust pady as needed
    
    label1 = tk.Label(convert_frame, text="Which graph problem would you like to convert to QUBO or Ising?", font=("Helvetica", 12, "bold"))
    label1.grid(row=0, column=0, columnspan=5, sticky='w', padx=(10, 0), pady=(0, 10))
    
    MaxCut_button = tk.Button(convert_frame, text="Maximum cut", command= lambda: my_f.convert_window(root, ct.MAXIMUM_CUT))
    MaxCut_button.grid(row=1, column=0, pady=5)
    
    tsp_button = tk.Button(convert_frame, text="Traveling Salesman Problem (TSP)", command= lambda: my_f.convert_window(root, "TSP"), state=tk.DISABLED)
    tsp_button.grid(row=1, column=1, pady=5)
    
    vertex_cover_button = tk.Button(convert_frame, text="Vertex Cover", command= lambda: my_f.convert_window(root, "VertexCover"), state=tk.DISABLED)
    vertex_cover_button.grid(row=1, column=2, pady=5)
    
    Next_button = tk.Button(convert_frame, text="Next problem", command= lambda: my_f.convert_window(root, "Next"), state=tk.DISABLED)
    Next_button.grid(row=1, column=3, pady=5)
    

def configuration_frame(root, row_position, cuda_info):
  
    config_frame = tk.Frame(root, bd=2, relief=tk.GROOVE, highlightthickness=2, highlightbackground="black")
    config_frame.grid(row=row_position, column=0, pady=10)  # Adjust pady as needed

    label1 = tk.Label(config_frame, text="Configure the system settings and then execute the QUBO solver!", font=("Helvetica", 12, "bold"))
    label1.grid(row=0, column=0, columnspan=5, sticky='w', padx=(10, 0), pady=(0, 10))
    
    new_setting_button = tk.Button(config_frame, text="New Settings", command= lambda: show_new_settings(cuda_info))
    new_setting_button.grid(row=1, column=1)
  
    edit_setting_button = tk.Button(config_frame, text="Edit Settings", command=lambda: show_new_settings(cuda_info,True))
    edit_setting_button.grid(row=1, column=3)
    

def execute_frame(root, row_position, cuda_info):
    
    # Bottom Frame
    exe_frame = tk.Frame(root, bd=2, relief=tk.GROOVE, highlightthickness=2, highlightbackground="black")
    exe_frame.grid(row=row_position, column=0, pady=10)  # Adjust pady as needed

    label1 = tk.Label(exe_frame, text="There Is at least one CUDA capable device on your system!", font=("Helvetica", 12, "bold"))
    label1.grid(row=0, column=0, columnspan=5, sticky='w', padx=(10, 0), pady=(0, 10))
    
    execute_setting_button = tk.Button(exe_frame, text="Execute QUBO", command=lambda: run_qubo(analyze_button))
    execute_setting_button.grid(row=1, column=1)
    
    analyze_button = tk.Button(exe_frame, text="Analyze the results", command=analyze, state='disabled')
    #analyze_button = tk.Button(root, text="Analyze the results", command=analyze)
    analyze_button.grid(row=1, column=3)
    

def main():

    
    # Convert Frame
    convert_Frame(root)
    
    # execute the device query to gather information about the GPU device
    cuda_info = my_f.run_deviceQuery()
     
    if cuda_info['num_devices'] > 0:
        
        # Configuration Frame
        row_position = 1
        configuration_frame(root, row_position, cuda_info)
        
        # Execute Frame
        row_position = 2
        execute_frame(root, row_position, cuda_info)
  
    
        
    else:
         tk.Label(root, text="Can Not Find any CUDA Capable device on your system!").grid(row=1, column=2)       
            
            



if __name__ == "__main__":
    root = tk.Tk()
    root.title("quantum inspired QUBO")
    root.geometry("600x300")
    
    # Configure the columns and rows to center the window
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    
    my_f.set_background_image(root)
    
    main()
    
    root.mainloop()