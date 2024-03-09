# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 17:23:32 2024

@author: bahman
"""

import subprocess

import tkinter as tk
from tkinter import filedialog
#from tkinter import ttk
from tkinter import messagebox
from tkinter import PhotoImage
import os
#from matplotlib.figure import Figure
#from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
#import pandas as pd
#from PIL import Image, ImageTk
import MaxCutToIsing as mx
import constant as ct



def set_background_image(window):
    """ 
    function to set the background image for a window
    """
    # Replace 'background_image.png' with the path to your image file
    image_path = 'nucleus02.png'
    
    # Create a PhotoImage object from the image file
    background_image = PhotoImage(file=image_path)

    # Set the image as the window background
    background_label = tk.Label(window, image=background_image)
    background_label.place(relwidth=1, relheight=1)

    # Ensure the image is not garbage collected
    window.image = background_image
    
    

def parse_device_query_log(log_file_path):
    """
    function to parse the device query's log and return some of its values
    """
    cuda_info = {
        'num_devices': 0,
        'total_global_memory': 0,
        'max_threads_per_block': 0,
        'total_shared_memory_per_block': 0
    }

    try:
        with open(log_file_path, 'r') as log_file:
            lines = log_file.readlines()

            for line in lines:
                # Check for lines containing relevant information
                if 'Detected' in line and 'CUDA Capable device(s)' in line:
                    cuda_info['num_devices'] = int(line.split()[1])
                elif 'Total amount of global memory' in line:
                    cuda_info['total_global_memory'] = int(line.split(':')[1].split()[0])
                elif 'Maximum number of threads per block' in line:
                    cuda_info['max_threads_per_block'] = int(line.split(':')[1].split()[0])
                elif 'Total amount of shared memory per block' in line:
                    cuda_info['total_shared_memory_per_block'] = int(line.split(':')[1].split()[0])

    except FileNotFoundError:
        print(f"Error: The specified log file '{log_file_path}' was not found.")
    except Exception as e:
        print(f"Error while parsing log file: {e}")

    return cuda_info

def run_deviceQuery():
    """
    function to run the deviceQuery.exe program and then return some of selected 
    query results    
    """
    exe_path = 'bin/deviceQuery.exe'
    output_file_path = 'bin/log.txt'
    print("Run deviceQuery")

    try:
        with open(output_file_path, 'w') as output_file:
            subprocess.run(exe_path, check=True, stdout=output_file, stderr=subprocess.STDOUT)
        print("deviceQuery executed! Output saved to log.txt")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
    except FileNotFoundError:
        print(f"Error: The specified .exe file '{exe_path}' was not found.")
        
    
    return parse_device_query_log(output_file_path)

###############################################################################

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
    

def browse_files(entry):
    filenames = filedialog.askopenfilenames()
    filenames = [relative_path_(filename) for filename in filenames]
        
    entry.delete(0, tk.END)
    entry.insert(0, "; ".join(filenames))


def browse_directory(entry):
    directory = filedialog.askdirectory()
    directory= relative_path_(directory)
    entry.delete(0, tk.END)
    entry.insert(0, directory)
    

def parse_convert_window_value(input_graph_matrix_entry, ising_or_qubo_option, output_dir_entry, penalty_entry):
    parsed_value={ct.GRAPH_FILE: input_graph_matrix_entry.get().split(";"),
              ct.EXECUTE_MODE: ising_or_qubo_option.get(),
              ct.OUTPUT_FOLDER: output_dir_entry.get(),
              ct.PENALTY: penalty_entry.get()}
    #.lstrip()
    parsed_value[ct.GRAPH_FILE] = [item.lstrip() for item in parsed_value[ct.GRAPH_FILE]]
    #print(parsed_value[ct.GRAPH_FILE])
    return parsed_value
        
      
def extract_input_file_name(path_to_input):
    file_names = [os.path.basename(path) for path in path_to_input]
    return file_names
    
def create_output_file_path(function, list_of_files, output_directory):
    base_folder = ''
    if output_directory != '':
        base_folder = output_directory + '\\'
    base_folder +=ct.ABBREVATION[function]
    list_of_output_file_Q = [base_folder + file + ".Q" for file in list_of_files]
    list_of_output_file_b = [base_folder + file + ".b" for file in list_of_files]
    
    return list_of_output_file_Q, list_of_output_file_b
                     

def convert_graph(function, input_graph_matrix_entry, ising_or_qubo_option, output_dir_entry, penalty_entry):
    parsed_value = parse_convert_window_value(input_graph_matrix_entry, ising_or_qubo_option, output_dir_entry, penalty_entry)
    
    list_of_input_file_name = extract_input_file_name(parsed_value[ct.GRAPH_FILE])
    list_of_output_file_Q, list_of_output_file_b = create_output_file_path(function, list_of_input_file_name, parsed_value[ct.OUTPUT_FOLDER])
    #print(list_of_output_file_Q)
    #print(list_of_output_file_b)
    if function == ct.MAXIMUM_CUT:
        #print(parsed_value)
        #mx.convertGraphToIsing()
        for i in range(len(list_of_output_file_Q)):
            print("Output file " +  list_of_output_file_Q[i] + "created.")
            #print(parsed_value[ct.GRAPH_FILE][i])
            #print(parsed_value[ct.EXECUTE_MODE])
            mx.convertGraphToIsing(list_of_output_file_Q[i], parsed_value[ct.GRAPH_FILE][i], convertMode=parsed_value[ct.EXECUTE_MODE])
            #convertGraphToIsing(out_path + file + "Q", in_filename=in_path + file, convertMode="QUBO")
        
    else:
        pass
    
    messagebox.showinfo("Information", "Conversion completed!")
    
def convert_window(root, function):
    new_window = tk.Toplevel(root)
    #new_window=tk.Tk()
    
    
    #label = tk.Label(new_window, text="This is a new window!")
    #label.pack()
    
    new_window.geometry("600x300")
    set_background_image(new_window)
    new_window.title("Convert Graph for" + function)
    
    tk.Label(new_window, text="Path to Matrix of Graph").grid(row=0, column=0)
    input_graph_matrix_entry = tk.Entry(new_window)
    input_graph_matrix_entry.grid(row=0, column=1)
    
    #browse_button = tk.Button(new_window, text="Browse", command=browse_file)
    browse_button = tk.Button(new_window, text="Browse", command=lambda: browse_files(input_graph_matrix_entry))
    browse_button.grid(row=0, column=2)
    
    #test_button = tk.Button(new_window, text="Find best Soulution (just Execute for small graph)", command=lambda: mx.test_maxCut_and_QUBO_of_graph(input_graph_matrix_entry))
    #test_button.grid(row=0, column=3)


    tk.Label(new_window, text="Execute Mode").grid(row=1, column=0)
    ising_or_qubo_var = tk.StringVar(new_window)
    ising_or_qubo_var.set("QUBO")  # default value
    ising_or_qubo_option = tk.OptionMenu(new_window, ising_or_qubo_var, "QUBO", "ISING")
    ising_or_qubo_option.grid(row=1, column=1)

       
    penality_need_function = []
    tk.Label(new_window, text="Enter a number for Penalty").grid(row=2, column=0)
    penalty_value_var = tk.IntVar()
    penalty_entry = tk.Entry(new_window, textvariable=penalty_value_var, state= tk.DISABLED)#, command=lambda: toggle_state_replica_exchange(num_replicas_var, max_temp_entry, exchange_attempts_entry))
    penalty_entry.grid(row=2, column=1)
    if function in penality_need_function:
        penalty_entry.config(state=tk.NORMAL)  
    



    tk.Label(new_window, text="Output folder").grid(row=3, column=0)
    output_dir_entry = tk.Entry(new_window)
    output_dir_entry.grid(row=3, column=1)

    browse_button_3 = tk.Button(new_window, text="Browse", command=lambda: browse_directory(output_dir_entry))
    browse_button_3.grid(row=3, column=2)

    save_button = tk.Button(new_window, text="Convert Graph for " + function , command=lambda: convert_graph(function, input_graph_matrix_entry, ising_or_qubo_var, output_dir_entry, penalty_entry))
    save_button.grid(row=4, column=1)
    return
    