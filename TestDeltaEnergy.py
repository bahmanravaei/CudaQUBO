# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 18:07:11 2023

@author: bahman
"""

import numpy as np
import random

def calculate_h(Q, x, i ):
    h = 0
    for j in range(len(x)):
        if (i!=j):
            h = h + Q[i,j] * x[j]
    
    h = 2* h + Q[i,i]
    return h

def compute_H(Q,x):
    H = np.zeros(len(x))
    for i in range(len(x)):
        H[i]=calculate_h(Q, x, i)
    return H

def update_H(H, Del_H, j):
    return H + Del_H[:, j] 
    


def delta_H(Q,x):
    Del_H = np.zeros((len(x),len(x)))
    for i in range (len(x)):
        for j in range(len(x)):
            if (i==j):
                Del_H[i,j] = 0
            else:
                Del_H[i,j]=Q[i,j]*(1-2*x[j])
    return 2 * Del_H

def update_delta_H(Del_H, x, j):
    for i in range(len(x)):
        Del_H[i,j]=-1 * Del_H[i,j]
    return Del_H


def energy(Q,x):
    return -np.dot(x, Q.dot(x))
    


def deltaEnergy(Q,x,i, H):
    delta_energy = -1 * (1 - 2 * x[i]) * H[i]
    return delta_energy

def energyD(E, Del_E):
    return E + Del_E
    
def qubo_delta_energy(Q, x, i, H):
    # Q: QUBO matrix
    # x: current binary configuration
    # i: index of the qubit to flip

    # Current energy before flip
    #energy_before = np.dot(x, Q.dot(x))
    energy_before =  energy(Q, x)
    # Flip the value of the qubit at index i
    x[i] = 1 - x[i]

    # Energy after flip
    #energy_after = np.dot(x, Q.dot(x))
    energy_after = energy(Q, x)
    
    # Flip the qubit back to the original configuration
    x[i] = 1 - x[i]

    # Calculate delta energy
    #delta_energy = 2 * Q[i, i] + 2 * sum(Q[i, j] * x[j] for j in range(len(x)) if j != i)
    
    delta_energy = -1 * (1 - 2 * x[i]) * calculate_h(Q,x,i)
    delta_energy = -1 * (1 - 2 * x[i]) * H[i]
    return delta_energy , energy_after - energy_before


def testEnergyEquality(x, Q, H):
    Flag = True
    for i_to_flip in range(len(x)):
        delta_energy1, delta_energy2 = qubo_delta_energy(Q, x, i_to_flip, H)
        print (str(delta_energy1) + " ? "+ str(delta_energy2) + " => " + str(delta_energy1==delta_energy2) )
        if (delta_energy1!=delta_energy2):
            Flag = False
    if (Flag!=True):
        print("ERROR")
    else:
        print("Correct")
    return Flag



def testDeltaH1(Q, x):
    Flag = True
    H = compute_H(Q,x)
    for j in range(len(x)):
        
      
        Del_H = delta_H(Q, x)
        x[j] = 1-x[j]
        #Del_H2 = delta_H(Q, x)
        H_new = compute_H(Q,x)
         
        
        #Del_H_real= H_new - H
        H_newU=update_H(H,Del_H,j)
        if(sum((H_newU-H_new)**2)!=0):
            Flag = False
            print(H_newU)
            print(H_new)
            print("---------------------")
        else:
            print(sum((H_newU-H_new)**2))
            print("+++++++++++++++++++++")
            
            
        #print(Del_H_real)
        x[j] = 1-x[j]
    if (Flag != True):
        print("ERROR")
    else:
        print("Correct")
    return Flag

def testEveryThing(Q,x, L, printLog = True):
    
    Flag = True
    
    E = energy(Q,x)
    Ed = E
    H = compute_H(Q,x)
    print(H)
    Del_H2 = delta_H(Q, x)
    print(Del_H2)
    for j in L:
        Logstr = "bit "
        Logstr  = Logstr + str(j) + " : " + str(x[j]) + " -> " + str(1-x[j]) 
        Del_E = deltaEnergy(Q,x,j, H)
        #Del_H = delta_H(Q,x)
        #print(Del_H)
        
        
        #print(Del_E)
        x[j] = 1-x[j]
        #H=update_H(H,Del_H,j)
        Logstr = Logstr +  "/\t E: " + str(E) + " -> "
        E = energy(Q,x)
        Logstr = Logstr + str(E)
        
        Ed = energyD(Ed,Del_E)
        Logstr = Logstr + " Ed: " + str(Ed)
        if (E!= Ed):
            Flag = False
            print("ERROR " + str(E) +" ? "+ str(Ed))
        
        H = update_H(H,Del_H2,j)
        #print(H)
        Del_H2 = update_delta_H(Del_H2, x, j)
        #print(Del_H2)
        #input("check delta_H")
        #H = compute_H(Q,x)
        if(printLog == True):
            print(Logstr)
    
    return Flag
        
        

###############################################################################
# Example usage:
Q = np.array([[5, 1, 2, 3], [1, 3, 4, 5], [2, 4, 1, 6], [3, 5, 6, 4]])  # QUBO matrix

x = np.array([1, 0, 1, 0])  # Example binary configuration

H = compute_H(Q,x)

testEnergyEquality(x, Q, H)

testDeltaH1(Q, x)

L = [0, 1, 2, 3, 3, 2, 1, 0]
L = random_numbers = [random.randint(0, len(x)-1) for _ in range(100)]
testEveryThing(Q,x, L)

###############################################################################
# Example 
Q = np.zeros((100,100))
filename = "g05_100.0Q"
data = np.loadtxt(filename, delimiter = " ", skiprows=1)
for item in data:
    Q[int(item[0])-1,int(item[1])-1]=item[2]



x = np.array([random.randint(0, 1) for _ in range(100)])  # Example binary configuration


L = random_numbers = [random.randint(0, len(x)-1) for _ in range(10000)]
testEveryThing(Q,x, L, printLog = True)

###############################################################################
# Example 
Q = np.zeros((5,5))
filename = "SimpleTest/QUBO/QUBOSimpleQ.txt"
data = np.loadtxt(filename, delimiter = " ", skiprows=1)
for item in data:
    Q[int(item[0])-1,int(item[1])-1]=item[2]



x = np.array([random.randint(0, 1) for _ in range(5)])  # Example binary configuration
x = np.array([0,0,0,0,0])
L = random_numbers = [random.randint(0, len(x)-1) for _ in range(10)]
L = [0,0,0,0,0,4, 1, 3]
testEveryThing(Q,x, L, printLog = True)

###############################################################################


for j in range(len(x)):
    x[j] = 1 - x[j]
    
    for i in range(len(x)):
        H[i] = H[i] - Del_H[i,j]
        #delta_energy1, delta_energy2 = qubo_delta_energy(Q, x, i, H)
        #print (str(delta_energy1) + " ? "+ str(delta_energy2) + " => " + str(delta_energy1==delta_energy2) )
    
    print("H and new_H")
    print(H)
    new_H = compute_H(Q,x)
    print (new_H)
    for i in range(len(x)):
        H[i] = H[i] + Del_H[i,j]
    x[j] = 1 - x[j]
    print("+++++++++++++++++++++")

   
    