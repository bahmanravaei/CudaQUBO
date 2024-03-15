# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 11:18:42 2023

@author: bahman
"""

import numpy as np
from itertools import product
import sys

#import tkinter as tk
#import my_utility_function as my_f
#from tkinter import filedialog
#from tkinter import ttk
#from tkinter import messagebox
#from tkinter import PhotoImage
#import os
#from matplotlib.figure import Figure
#from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
#import pandas as pd
#from PIL import Image, ImageTk
#import MaxCutToIsing as maxCut



def max_cut_to_qubo(adjacency_matrix):
    """
    Convert a graph represented by its adjacency matrix to a QUBO model for maximum cut.

    Parameters:
    - adjacency_matrix (numpy.ndarray): The adjacency matrix of the graph.

    Returns:
    - Q (numpy.ndarray): The QUBO matrix.
    """
    # Get the number of nodes in the graph
    num_nodes = len(adjacency_matrix)

    # Initialize the QUBO matrix
    Q = np.zeros((num_nodes, num_nodes))

    # Set the QUBO matrix based on the edge weights in the graph
    for i in range(num_nodes):
        R=0
        for j in range(num_nodes):
            R = R + adjacency_matrix[i,j]
            Q[i, j] = - adjacency_matrix[i, j]
        
        #print(R)
        Q[i, i] = R - adjacency_matrix[i,i]
        
    return Q

def max_cut_to_ising(adjacency_matrix):
    """
    Convert a graph represented by its adjacency matrix to an Ising model for maximum cut.

    Parameters:
    - adjacency_matrix (numpy.ndarray): The adjacency matrix of the graph.

    Returns:
    - h (numpy.ndarray): The linear terms of the Ising model.
    - J (numpy.ndarray): The quadratic terms of the Ising model.
    """
    # Get the number of nodes in the graph
    num_nodes = len(adjacency_matrix)

    # Initialize the Ising model parameters
    h = np.zeros(num_nodes)
    J = np.zeros((num_nodes, num_nodes))

    # Set the linear terms (h) based on the degree of each node
    for i in range(num_nodes):
        h[i] = np.sum(adjacency_matrix[i, :])

    # Set the quadratic terms (J) based on the edge weights in the graph
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adjacency_matrix[i, j] != 0:
                J[i, j] = -adjacency_matrix[i, j]
                J[j, i] = -adjacency_matrix[i, j]

    return h, J


#--------------------------------------------------
def write_A_b_ToFile(J, outFilenameJ, h = [], outFilenameH=""):
    numberofedge=0
    for i in range(len(J)):
        for j in range(len(J)):
            if J[i][j] != 0:
                numberofedge=numberofedge+1
                
    print(numberofedge)
    
    f = open(outFilenameJ, "w")
    f.write(str(len(J))+ " " + str(numberofedge)+"\n")
    
    
    for i in range(len(J)):
        for j in range(len(J)):
            if J[i][j]!=0:
                tempStr=str(i+1)+ " " + str(j+1) + " " + str(J[i][j]) + "\n"
                f.write(tempStr)        
    
    f.close()
    
    if(len(h)!=0):
        f = open(outFilenameH, "w")
        for i in range(len(h)):
            tempStr = str(h[i])+ "\n"
            f.write(tempStr)
        f.close()

#--------------------------------------------------
def fileGraphToArray(filename):
    #print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    #print(filename)
    #print("len file name: " + str(len(filename)))
    try:
        data = np.loadtxt(filename, delimiter = " ", skiprows=1)
        #data=data-1
        data[:, 0:2]-=1           # Start node index from 0: (1, n) => (0, n-1)
        
        maxNode = int(np.max(data[:, :2])+1)    # Finding the number of nodes
        Arr = np.zeros((maxNode, maxNode))
        for i in range(data.shape[0]):
            Arr[int(data[i,0]),int(data[i,1])]=int(data[i,2])
            Arr[int(data[i,1]),int(data[i,0])]=int(data[i,2])
    except Exception as e:
        
        print(f"Error reading file: {e}")
        #sys.exit()
    
    
    return Arr
    
#--------------------------------------------------
def testEnergyEvaluation(Q, X, Flag, b = np.array([])):
    ## Flag == True => Q is in form of QUBO , Flag == False  => Q is adjacency Matrix
    if Flag == True:
        XBar=X
    else:
        XBar=1-X

    E = np.dot(np.dot(X, Q),XBar)

    if(len(b)!=0):
        #print("\t\t\t\t\t Energy: " + str(E)+ " + " +str(np.dot(X,b)))

        E = E + np.dot(X,b)
        #print("\t\t\t testEnergyEvaluat...: " + str(E))
    
    #print(X)
    #print(XBar)
    
    return E

#--------------------------------------------------
def generate_coin_toss_outcomes(n, isingFlag):
    """
    Generate all possible outcomes of tossing n coins.

    Parameters:
    - n (int): Number of coins.

    Returns:
    - list: List of all possible outcomes.
    """
    coin_outcomes = [0, 1]
    if isingFlag == True:
        coin_outcomes = [-1, 1]
    
    all_outcomes = list(product(coin_outcomes, repeat=n))
    np_outcome=np.array(all_outcomes)
    return np_outcome

#--------------------------------------------------
def test_for_all_bits_configuration(Q, n=5, Flag=False, isingFlag=False , B = []):
    """
    
    Parameters
    ----------
    Q : Coupling or Adjacency Matrix
        if Flag == True => Q is in form of QUBO , Flag == False  => Q is adjacency Matrix
    n : int, optional
        size of graph or matrix. The default is 5.
    Flag : bool, optional
        Qubo (True) or graph (False). The default is False.
    isingFlag : bool, optional
        Qubo or Ising. The default is False.
    B : 1-D List, optional
        The bias array. The default is [].

    Returns
    -------
    L : List
        List of all bit configuration and their energy.

    """
    ## 
    #n=5
    all_X=generate_coin_toss_outcomes(n,isingFlag)
    L=[]
    optimaValue=0
    if(Flag):
        if(isingFlag==True):
            print("Best bits configuration and Energy, when the Q is adjacency Matrix and Ising problem:")
        else:
            print("Best bits configuration and Energy, when the Q is adjacency Matrix (QUBO problem):")
    else:
        print("Best bits configuration and Energy, when the Q is adjacency Matrix of graph:")
    
    for item in all_X:
        energy_value=testEnergyEvaluation(Q,item, Flag, b = B)
        if (optimaValue<=energy_value):
            print("\t" , item, energy_value)
            optimaValue=energy_value
        L.append((item,energy_value))
    
    return L
    
#--------------------------------------------------
def convertGraphToIsing(out_filename_A, in_filename="", graph_adjacency_matrix=[], out_filename_b="", convertMode="Ising"):
    if (in_filename!=""):
        graph_adjacency_matrix = fileGraphToArray(in_filename)
        
    elif(len(graph_adjacency_matrix)==0):
        print("****   ERROR   ****")
        return 
            
        

    if (convertMode == "QUBO"):
        # Convert to QUBO model
        A = max_cut_to_qubo(graph_adjacency_matrix)
    elif(convertMode=="Ising"):
# Convert to Ising model
        b, A = max_cut_to_ising(graph_adjacency_matrix)
        if(out_filename_b==""):
            for i in range(len(b)):
                A[i,i]=b[i]
            convertMode = "QUBO"
    
    if(convertMode == "QUBO"):
        write_A_b_ToFile(A, out_filename_A)
    elif (convertMode=="Ising"):
        write_A_b_ToFile(A,out_filename_A , b,out_filename_b)
        
        
            
#--------------------------------------------------
###############################################################################   
filenames = ["ising2.5-100_5555.txt", "ising2.5-100_5555_edited.bq"]
#filename=["g05_100.0","g05_100.1","g05_100.2","g05_100.3",
#          "g05_100.4","g05_100.5","g05_100.6","g05_100.7",
#          "g05_100.8","g05_100.9"]

in_path="testFile/"
out_path = "testFile/"
convertMode="QUBO"
def convert_graph_files_to_Ising(filenames, in_path, out_path, convertMode):
    
    
    for file in filenames:
        convertGraphToIsing(out_path + file + "Q", in_filename=in_path + file, convertMode = convertMode)
        
    
    #"Files/maxCutAg05_100.0.txt", "Files/maxCutBg05_100.0.txt"            

###############################################################################
## Codes for after executing the C code and findig the best configuration
fileNameGraph = "C:/Users/bahman/source/repos/CudaQUBO/SimpleTest/maxCut_n10_e18_res74.csv"
best_cut_file = "C:/Users/bahman/source/repos/CudaQUBO/SimpleTest/QUBO/latticeBest.csv"

fileNameGraph = "C:/Users/bahman/source/repos/CudaQUBO/SimpleTest/maxCut_n20_e38_res271.csv"
best_cut_file = "C:/Users/bahman/source/repos/CudaQUBO/SimpleTest/QUBO/latticeBest.csv"

fileNameGraph = "C:/Users/bahman/source/repos/CudaQUBO/maxCutGraph/g05_100.4"
fileNameGraph = "C:/Users/bahman/source/repos/CudaQUBO/maxCutGraph/ising2.5-100_5555.bq.txt"
def test_energy_of_a_configuration(fileNameGraph, cut_file, Flag):
    """

    Parameters
    ----------
    fileNameGraph : Path to graph file
        DESCRIPTION.
    cut_file : Path to a configuration of bits
        DESCRIPTION.
    Flag :  bool
        Flag == True => Q is in form of QUBO , Flag == False  => Q is adjacency Matrix
    Returns
    -------
    Find energy.

    """
    #fileNameGraph = "maxCutGraph/g05_100.0"
    graph_adjacency_matrix = fileGraphToArray(fileNameGraph)
    Q=graph_adjacency_matrix
    
    aCut = np.loadtxt(cut_file, delimiter = ",")
    """
    """
    E=testEnergyEvaluation(Q, aCut, Flag)
    return E
    ## Flag == True => Q is in form of QUBO , Flag == False  => Q is adjacency Matrix

###############################################################################
number_of_replica = 10
list_of_cut_file = ["Excel/Initlattice"+str(i)+".csv" for i in range(number_of_replica)]
list_of_cut_file.append("Excel/latticeBest.csv")
[list_of_cut_file.append("Excel/latticeFinal"+str(i)+".csv") for i in range(number_of_replica)]

fileNameGraph = "testFile/ising2.5-100_5555_with_loop.gr"
fileName_Q = "testFile/ising2.5-100_5555_with_loop.gr.Q"
fileNameGraph2 = "testFile/ising2.5-100_5555_with_loop.gr"
fileName_Q2 = "testFile/ising2.5-100_5555_with_loop.gr.Q"

fileName_Q_or_graph = [fileName_Q, fileNameGraph, fileName_Q2, fileNameGraph2]

a_cut = "testFile/bestSolution.csv"
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
        for file_of_matrix in fileName_Q_or_graph:
            for Flag in FlagList:
                E = test_energy_of_a_configuration(file_of_matrix, filename, Flag)
                print(file_of_matrix, filename, Flag, E)
        print("___________________________________________")


###############################################################################
#                  Code for test the correctness of program
# Example usage:
# Define the adjacency matrix of the graph
def test4():
    graph_adjacency_mat = np.array([[0, -10, 5, 8, -6],
                                       [-10, 0, -3, 0, 4],
                                       [5, -3, 0,  -5, 0],
                                       [8, 0, -5, 0, -8],
                                       [-6, 4, 0, -8, 0]])
    
    out_filename_A = "SimpleTest/IsingSimpleA00.txt"
    convertGraphToIsing(out_filename_A, graph_adjacency_matrix=graph_adjacency_mat, convertMode="Ising")
    
    out_filename_A = "SimpleTest/IsingSimpleA01.txt"
    filename_b = "SimpleTest/IsingSimpleb01.txt"
    convertGraphToIsing(out_filename_A, graph_adjacency_matrix=graph_adjacency_mat, out_filename_b=filename_b, convertMode="Ising")
    
    
    
    out_filename_Q = "SimpleTest/QUBOSimpleQ.txt"
    convertGraphToIsing(out_filename_Q, graph_adjacency_matrix=graph_adjacency_mat, convertMode="QUBO")

###############################################################################

# Example usage:
# Define the adjacency matrix of the graph
def test_maxCut_and_QUBO_of_graph(file_name, print_flag=False):
    """
    Test All bits configuration under different condition (Graph, Qubo, Ising)
    and print the best energy during the evaluation.
    It seems it works for all cases, !!!!!!     EXCEPT ONE CASE     !!!!!
    Parameters
    ----------
    file_name : string
        Path to a file that contains the adjacency of a graph.
        
    print_flag: bool
        flag for print (True) 
    """
    
    
    graph_adjacency_mat = fileGraphToArray(file_name)
    #return graph_adjacency_mat
    n = len(graph_adjacency_mat)            #size of adjacency matrix (n * n)
    resMaxCut=test_for_all_bits_configuration(graph_adjacency_mat, n , Flag=False)   # It is a maximization problem
    if(print_flag==True):
        print("All bit configuration max Cut (As a maximization problem): ")
        print(resMaxCut)
    
    # Convert to QUBO model
    Q = max_cut_to_qubo(graph_adjacency_mat)
    #print(Q)
    resQUBO=test_for_all_bits_configuration(Q, n, Flag=True)                         # It is a maximization problem
    if(print_flag==True):
        print("All bit configuration max Cut (As a maximization problem): ")
        print(resMaxCut)

    
    # Convert to Ising model
    h, J = max_cut_to_ising(graph_adjacency_mat)
    
    resIsing_1=test_for_all_bits_configuration(J, n, Flag=True, isingFlag=False, B = h)
    resIsing_1=test_for_all_bits_configuration(J, n, Flag=True, isingFlag=True, B = h)
    
    
    # For testing energy in Ising model
    for i in range(len(h)):
        J[i,i]=h[i]
    
    resIsing_2=test_for_all_bits_configuration(J, n, Flag=True, isingFlag=True)      # It is a maximization problem
       

###############################################################################


