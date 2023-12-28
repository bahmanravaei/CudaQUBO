# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 11:18:42 2023

@author: bahman
"""

import numpy as np
from itertools import product

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
        Q[i, i] = R
        
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
    data = np.loadtxt(filename, delimiter = " ", skiprows=1)
    data=data-1
    maxNode = int(np.max(data)+1)
    Arr = np.zeros((maxNode, maxNode))
    for i in range(data.shape[0]):
        Arr[int(data[i,0]),int(data[i,1])]=1
        Arr[int(data[i,1]),int(data[i,0])]=1
    
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
        print("\t\t\t\t\t Energy: " + str(E)+ " + " +str(np.dot(X,b)))

        E = E + np.dot(X,b)
        print("\t\t\t testEnergyEvaluation: " + str(E))
    
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
def testForallScenarios(Q, n, Flag=False, isingFlag=False , B = []):
    ## Flag == True => Q is in form of QUBO , Flag == False  => Q is adjacency Matrix
    n=5
    all_X=generate_coin_toss_outcomes(n,isingFlag)
    L=[]
    optimaValue=0
    for item in all_X:
        res=testEnergyEvaluation(Q,item, Flag, b = B)
        if (optimaValue<=res):
            print(item, res)
            optimaValue=res
        L.append((item,res))
    
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
filename=["g05_100.0","g05_100.1","g05_100.2","g05_100.3",
          "g05_100.4","g05_100.5","g05_100.6","g05_100.7",
          "g05_100.8","g05_100.9"]

in_path="Files/"
out_path = "maxCut/"
for file in filename:
    convertGraphToIsing(out_path + file + "Q", in_filename=in_path + file, convertMode="QUBO")
    

"Files/maxCutAg05_100.0.txt", "Files/maxCutBg05_100.0.txt"            

###############################################################################
## Codes for after executing the C code for Ising model
fileNameGraph = "maxCutGraph/g05_100.0"
graph_adjacency_matrix = fileGraphToArray(fileNameGraph)
Q=graph_adjacency_matrix
filename="Files/latticeFinal.csv"
aCut = np.loadtxt(filename, delimiter = ",")

bestFoundCut=np.array(
      [1., 1., 1., 1., 0., 1., 1., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 1., 0., 1., 1., 1., 0., 1., 1., 0., 1., 1., 1., 0., 1., 0.,
       1., 0., 1., 0., 1., 0., 1., 1., 1., 0., 1., 0., 1., 0., 0., 1., 1.,
       0., 1., 0., 0., 0., 1., 1., 1., 0., 0., 1., 0., 0., 0., 1., 0., 1.,
       1., 1., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 1., 1., 0.,
       1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1.])

E=testEnergyEvaluation(Q, bestFoundCut, False)
## Flag == True => Q is in form of QUBO , Flag == False  => Q is adjacency Matrix

###############################################################################

filename="Files/brock200-1.mtx"
filename="Files/g05_100.0"


###############################################################################
#                  Code for test the correctness of program
# Example usage:
# Define the adjacency matrix of the graph
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
graph_adjacency_mat = np.array([[0, -10, 5, 8, -6],
                                   [-10, 0, -3, 0, 4],
                                   [5, -3, 0,  -5, 0],
                                   [8, 0, -5, 0, -8],
                                   [-6, 4, 0, -8, 0]])

resMaxCut=testForallScenarios(graph_adjacency_mat, 5, Flag=False)   # It is a maximization problem

# Convert to QUBO model
Q = max_cut_to_qubo(graph_adjacency_mat)
resQUBO=testForallScenarios(Q,5, Flag=True)                         # It is a maximization problem


# Convert to Ising model
h, J = max_cut_to_ising(graph_adjacency_mat)
resIsing_1=testForallScenarios(J,5, Flag=True, isingFlag=False, B = h)
resIsing_1=testForallScenarios(J,5, Flag=True, isingFlag=True, B = h)


# For testing energy in Ising model
for i in range(len(h)):
    J[i,i]=h[i]

resIsing_2=testForallScenarios(J,5, Flag=True, isingFlag=True)      # It is a maximization problem

