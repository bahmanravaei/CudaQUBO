# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 18:03:36 2023

@author: bahman
"""

filename = 'WInit.csv'
outFilename = 'W.txt'
List=[]
with open(filename) as file_object:
    for line in file_object:
        line=line[0:len(line)-1]
        L=line.split(',')
        List.append(L)

numberofedge=0
for i in range(len(List)):
    for j in range(len(List)):
        if List[i][j] == '1':
            numberofedge=numberofedge+1
            
print(numberofedge)

f = open(outFilename, "w")
f.write(str(len(List))+ " " + str(numberofedge)+"\n")


for i in range(len(List)):
    for j in range(len(List)):
        if List[i][j]!='0':
            tempStr=str(i+1)+ " " + str(j+1) + " " + List[i][j] + "\n"
            f.write(tempStr)        

f.close()
                   

