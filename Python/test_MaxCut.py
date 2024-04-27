# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 11:12:18 2024

@author: bahman
"""

import MaxCutToIsing as maxCut
import constant as ct
import numpy as np
import pandas as pd
import csv

###############################################################################
number_of_replica = 10
list_of_cut_file = []
list_of_cut_file = ["Excel/Initlattice"+str(i)+".csv" for i in range(number_of_replica)]
[list_of_cut_file.append("Excel/latticeFinal"+str(i)+".csv") for i in range(number_of_replica)]
list_of_cut_file.append("Excel/latticeBest.csv")


fileNameGraph = "testFile/ising2.5-100_5555_with_loop.gr"
fileName_Q = "testFile/ising2.5-100_5555_with_loop.gr.Q"
fileNameGraph2 = "testFile/ising2.5-100_5555_with_loop.gr"
fileName_Q2 = "testFile/ising2.5-100_5555_with_loop.gr.Q"

fileName_Q_or_graph = [fileName_Q, fileNameGraph, fileName_Q2, fileNameGraph2]

a_cut = "testFile/bestSolution.csv"
list_of_cut_file.append(a_cut)
a_cut = "testFile/bestSolutionTemp.csv"
list_of_cut_file.append(a_cut)


fileName_Q = "testFile/maxCut_n_5_e_res_8Q.txt"
fileNameGraph = "testFile/maxcut_simple_graph_q_in_book.q"

fileName_Q = "testFile/maxCutmaxCut_n10_e18_res74.csv.Q"
fileNameGraph = "testFile/maxCut_n10_e18_res74.csv"

fileName_Q = "testFile/maxCutmaxCut_n20_e38_res271.csv.Q"
fileNameGraph = "testFile/maxCut_n20_e38_res271.csv"

fileName_Q = "testFile/g05_100.0Q"
fileNameGraph = "testFile/g05_100.0"

FlagList = [True, False]

fileName_Q_or_graph = [fileName_Q, fileNameGraph]

def test_calculate_energy_of_cut_list(fileName_Q_or_graph, list_of_cut_file, FlagList):
    # Flag == True => Q is in form of QUBO , Flag == False  => Q is adjacency Matrix
    print("|Trace\t|\t\tQ\t\t|\t\tG\t\t|")
    print("___________________________________________")
    print("|\t\t|\tT\t|\tF\t|\tT\t|\tF\t|")
    print("___________________________________________")
    
    for filename in list_of_cut_file:
        #print(filename)
        for file_of_matrix in fileName_Q_or_graph:
            #print(file_of_matrix)
            for Flag in FlagList:
                E = maxCut.test_energy_of_a_configuration(file_of_matrix, filename, Flag)
                #print(Flag, E/2)
                print(E, end = " ")
        print("\n___________________________________________")


all_cases = {}
all_cases[10]=["bestSolution10.csv", "maxCutising2.5-100_5555_sub_graph11.bq.Q", "ising2.5-100_5555_sub_graph11.bq", "gurobi10.csv"]
all_cases[20]=["bestSolution20.csv", "maxCutising2.5-100_5555_sub_graph21.bq.Q", "ising2.5-100_5555_sub_graph21.bq", "gurobi20.csv"]
all_cases[30]=["bestSolution30.csv", "maxCutising2.5-100_5555_sub_graph31.bq.Q", "ising2.5-100_5555_sub_graph31.bq", "gurobi30.csv"]
all_cases[40]=["bestSolution40.csv", "maxCutising2.5-100_5555_sub_graph41.bq.Q", "ising2.5-100_5555_sub_graph41.bq", "gurobi40.csv"]
all_cases[50]=["bestSolution50.csv", "maxCutising2.5-100_5555_sub_graph51.bq.Q", "ising2.5-100_5555_sub_graph51.bq", "gurobi50.csv"]
all_cases[60]=["bestSolution60.csv", "maxCutising2.5-100_5555_sub_graph61.bq.Q", "ising2.5-100_5555_sub_graph61.bq", "gurobi60.csv"]
all_cases[70]=["bestSolution70.csv", "maxCutising2.5-100_5555_sub_graph71.bq.Q", "ising2.5-100_5555_sub_graph71.bq", "gurobi70.csv"]
all_cases[80]=["bestSolution80.csv", "maxCutising2.5-100_5555_sub_graph81.bq.Q", "ising2.5-100_5555_sub_graph81.bq", "gurobi80.csv"]
all_cases[90]=["bestSolution90.csv", "maxCutising2.5-100_5555_sub_graph91.bq.Q", "ising2.5-100_5555_sub_graph91.bq", "gurobi90.csv"]
all_cases[100]=["bestSolution100.csv", "ising2.5-100_5555_with_loop.gr.Q", "ising2.5-100_5555_with_loop.gr", "gurobi100.csv"]
all_cases[101]=["bestSolution100.csv", "maxCutising2.5-100_5555_sub_graph101.bq.Q", "ising2.5-100_5555_sub_graph101.bq", "gurobi100.csv"]

output_path = "C:/Users/bahman/source/repos/CudaQUBO/maxCutGraph/convertedfile/"

FlagList = [True, False]


for problem_key in range(10,101,10):
    print(f"\n\n___________________{problem_key}_____________________")
    fileName_Q = output_path + all_cases[problem_key][1]
    fileNameGraph = output_path + all_cases[problem_key][2]
    fileName_Q_or_graph = [fileName_Q, fileNameGraph]
    list_of_cut_file = [output_path + all_cases[problem_key][0], output_path + all_cases[problem_key][3]]
    test_calculate_energy_of_cut_list(fileName_Q_or_graph, list_of_cut_file, FlagList)

######################################################################################
# Energy of Neighbour configurations of a Cut:

def find_neighbours_of_a_config_solution(aCut):
    
    
    modified_arrays = []

    # Iterate over each index in the original array
    for i in range(len(aCut)):
        # Copy the original array to modify
        modified_array = np.copy(aCut)
    
        # Toggle the value of the item at index i
        modified_array[i] = 1 - modified_array[i]
    
        # Append the modified array to the list
        modified_arrays.append(modified_array)
    return modified_arrays    

def calculate_energy_of_list_of_config(modified_arrays,Q, is_adjacency_matrix_flag):
    list_energy = []
    for i in range(len(modified_arrays)):
        energy = maxCut.testEnergyEvaluation(Q, modified_arrays[i], is_adjacency_matrix_flag)
        list_energy.append(energy)
    return list_energy
    
    
    
def energy_neighbour_configurations(cut_file, Q, is_adjacency_matrix_flag):
    aCut = np.loadtxt(cut_file, delimiter = ",")

    modified_arrays = find_neighbours_of_a_config_solution(aCut)
    
    
    
    list_energy = calculate_energy_of_list_of_config(modified_arrays,Q, is_adjacency_matrix_flag)
    energy = maxCut.testEnergyEvaluation(Q, aCut, is_adjacency_matrix_flag)
    list_energy.insert(0, energy)
    return np.array(list_energy)

def all_configs(path_for_all):
    L = []
    for cut_file in path_for_all:
        aCut = np.loadtxt(cut_file, delimiter = ",")
        
        L.append(aCut)
        array_2d = np.vstack(L)
    return array_2d
    
array_2d=all_configs(path_for_all)

###############################################################################

def generate_path_of_all_files(dir_path, file_prefix, number_of_files):
    files_path = []
    for i in range(number_of_files):
        files_path.append(dir_path+file_prefix+str(i)+".csv")
    return files_path

##############################################################################

dir_path = "C:/Users/bahman/source/repos/CudaQUBO/SimpleTest/QUBO/"
file_prefix = "bestConfig"
number_of_files = 58
path_for_all = generate_path_of_all_files(dir_path, file_prefix, number_of_files)



best_cut_file = "testFile/bestSolution.csv"
best_cut_file = "C:/Users/bahman/source/repos/CudaQUBO/maxCutGraph/convertedfile/bestSolution60.csv"
path_for_all.insert(0, best_cut_file)

is_adjacency_matrix_flag = False
Q_matrix_file = "testFile/ising2.5-100_5555_with_loop.gr.Q"
Q_matrix_file = "C:/Users/bahman/source/repos/CudaQUBO/maxCutGraph/convertedfile/ising2.5-100_5555_sub_graph61.bq"

Q = maxCut.fileGraphToArray(Q_matrix_file)
df = pd.DataFrame()

i = -1
for a_cut_file in path_for_all:
    A = energy_neighbour_configurations(a_cut_file, Q, is_adjacency_matrix_flag)
    column_name = f"A_{i}"  # Naming convention for columns
    df[column_name] = A.flatten()
    i = i+1
    
excel_filename = "C:/Users/bahman/source/repos/CudaQUBO/SimpleTest/QUBO/arrays_data60.xlsx"  # Choose a filename
df.to_excel(excel_filename, index=False)
    
##############################################################################



def generate_smaller_problems(file_name,size_of_smaller_problem): 
    data = np.loadtxt(file_name, delimiter = " ", skiprows=2)
    subgraph = data[(data[:, 0] < size_of_smaller_problem) & (data[:, 1] < size_of_smaller_problem)]
    return subgraph

def save_graph(output_file, graph, number_of_vertex):
    f = open(output_file, "w")
    f.write(str(number_of_vertex-1) + " "+ str(len(graph))+"\n")
    for i in range(len(graph)):
        line =  f"{int(graph[i,0])} {int(graph[i,1])} {(graph[i,2])}\n"
        f.write(line)
    f.close()

graph_file = "C:/Users/bahman/source/repos/CudaQUBO/maxCutGraph/ising2.5-100_5555.bq"    
output_path = "C:/Users/bahman/source/repos/CudaQUBO/maxCutGraph/ising2.5-100_5555_sub_graph"
for graph_size in range(11, 102, 10):
    newGraph = generate_smaller_problems(graph_file, graph_size)
    output_file_path = output_path+str(graph_size)+".bq"
    print(output_file_path)
    save_graph(output_file_path, newGraph, graph_size)
    

############################################################
def generate_csv(L, N, filename):
    # Create a list of zeros with length N
    values = [0] * N
    
    # Set the elements at indices specified by list L to 1
    for index in L:
        if index <= N:
            values[index-1] = 1
    
    # Write the values to a CSV file
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(values)


solution={}
solution[10] = [1, 2, 3, 4, 5, 6, 7, 8]
solution[100] = [9, 10, 11, 12, 16, 17, 18, 19, 24, 26, 27, 28, 30, 35, 37, 38, 40, 46, 50, 52, 53, 58, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 74, 76, 78, 81, 83, 84, 86, 88, 90, 91, 92, 98, 99, 100]
#solution[100] = [9, 10, 11, 12, 16, 17, 18, 19, 24, 26, 27, 28, 30, 35, 37, 38, 40, 46, 50, 52, 53, 58, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 74, 76, 78, 81, 83, 84, 86, 88, 90, 91, 92, 98, 99, 100]
energy={}
energy[11] = -91142
energy[100] = -2460049
for key, value in solution.items():
    a_solution = value
    output_path = "C:/Users/bahman/source/repos/CudaQUBO/maxCutGraph/convertedfile/bestSolution"
    file_path = f"{output_path}{key}.csv"
    print(file_path)
    generate_csv(a_solution, key, file_path)

#####################################################################################################

all_cases = {}
all_cases[10]=["bestSolution10.csv", "maxCutising2.5-100_5555_sub_graph11.bq.Q", "ising2.5-100_5555_sub_graph11.bq"]
all_cases[20]=["bestSolution20.csv", "maxCutising2.5-100_5555_sub_graph21.bq.Q", "ising2.5-100_5555_sub_graph21.bq"]
all_cases[30]=["bestSolution30.csv", "maxCutising2.5-100_5555_sub_graph31.bq.Q", "ising2.5-100_5555_sub_graph31.bq"]
all_cases[40]=["bestSolution40.csv", "maxCutising2.5-100_5555_sub_graph41.bq.Q", "ising2.5-100_5555_sub_graph41.bq"]
all_cases[50]=["bestSolution50.csv", "maxCutising2.5-100_5555_sub_graph51.bq.Q", "ising2.5-100_5555_sub_graph51.bq"]
all_cases[60]=["bestSolution60.csv", "maxCutising2.5-100_5555_sub_graph61.bq.Q", "ising2.5-100_5555_sub_graph61.bq"]
all_cases[70]=["bestSolution70.csv", "maxCutising2.5-100_5555_sub_graph71.bq.Q", "ising2.5-100_5555_sub_graph71.bq"]
all_cases[80]=["bestSolution80.csv", "maxCutising2.5-100_5555_sub_graph81.bq.Q", "ising2.5-100_5555_sub_graph81.bq"]
all_cases[90]=["bestSolution90.csv", "maxCutising2.5-100_5555_sub_graph91.bq.Q", "ising2.5-100_5555_sub_graph91.bq"]
all_cases[100]=["bestSolution100.csv", "ising2.5-100_5555_with_loop.gr.Q", "ising2.5-100_5555_with_loop.gr"]
all_cases[101]=["bestSolution100.csv", "maxCutising2.5-100_5555_sub_graph101.bq.Q", "ising2.5-100_5555_sub_graph101.bq"]

output_path = "C:/Users/bahman/source/repos/CudaQUBO/maxCutGraph/convertedfile/"

def test_calculate_entergy_for_a_group_of_scenarios(all_cases, output_path):
    
    for key, value in all_cases.items():
        if key !=20:
            continue
        fileName_Q = [output_path + value[1], output_path + value[2]]
        a_cut_file = [output_path + value[0]]
        
        FlagList = [True, False]
        print(key, fileName_Q, a_cut_file, FlagList, sep="\n---------------------\n")
        
        test_calculate_energy_of_cut_list(fileName_Q, a_cut_file, FlagList)
        
    
test_calculate_entergy_for_a_group_of_scenarios(all_cases, output_path)
###########

def to_matlab_format(Q):
    output_str = ""
    for i in range(len(Q)):
        for j in range()

def create_Q_matrix_for_matlab(all_cases, index=[], output_path=""):
    for i in index:
        print(output_path + all_cases[i][1])
        graph_adjacency_matrix = maxCut.fileGraphToArray(output_path + all_cases[i][1])
        print(graph_adjacency_matrix)
        
create_Q_matrix_for_matlab(all_cases, [10], output_path)