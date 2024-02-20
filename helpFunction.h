#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <filesystem>
#include <sys/stat.h>
#include "ising.h"
#include "constant.h"

using namespace std;



// The printVector function takes in an array of integers (X) and its length (lenX) as parameters. 
// This function displays the contents of the array
void printVector(int* X, int lenX) {
    for (int i = 0; i < lenX; i++) {
        cout << X[i] << "\t";
    }
    cout << endl;
}



// This function is for testing purpose and can be deleted in final version of code
bool testMatrix(double** X, double* aRow, int rowIndex, int sizeOfRow) {
    int tempInt;
    for (int i = 0; i < sizeOfRow; i++) {
        if (X[rowIndex][i] != aRow[i]) {
            cout << "__________________ERROR__________________" << endl;
            cout << "rowIndex: " << rowIndex << " i: " << i << endl;
            cout << "X[rowIndex]: " << X[rowIndex] << " aRow:" << aRow << endl;
            cin >> tempInt;
            return false;
        }
    }
    return true;
}

void testPointer(int** X, int num_replicas) {
    for (int i = 0; i < num_replicas; i++)
        cout << X[i] << "\t";
    cout << endl;
}





unsigned int nextPowerOf2(unsigned int N) {
    N--; // Ensure that if N is a power of 2, the result is not doubled
    N |= N >> 1;
    N |= N >> 2;
    N |= N >> 4;
    N |= N >> 8;
    N |= N >> 16;
    return N + 1;
}

void printX(int* X, int lenX, string str) {
	std::cout << str << " : [";
	for (int i = 0; i < lenX; i++)
		std::cout << X[i] << " ";
	std::cout << "] " << endl;
}

void printH(double* H, int lenH, string str) {
	std::cout << str <<" : [";
	for (int i = 0; i < lenH; i++)
		std::cout << H[i] << " ";
	std::cout << "] " << endl;
}


void printAllH(double** H, int row, int col) {
    for (int i = 0; i < row; i++) {
        string str = "H for replica " + to_string(i);
        printH(H[i], col, str);
    }
}

double* convert2Dto1D(double** array2D, int row, int col) {
    double* array1D = new double[row * col];
    int index = 0;
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            array1D[index] = array2D[i][j];
            index++;
        }
    }
    return array1D;
}

int* convert_int_2Dto1D(int** array2D, int row, int col) {
    int* array1D = new int[row * col];
    int index = 0;
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            array1D[index] = array2D[i][j];
            index++;
        }
    }
    return array1D;
}

double** convert1Dto2D(double* array1D, double** array2D, int row, int col) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            array2D[i][j] = array1D[i * row + j];
        }
    }
    return array2D;
}

void fill1Darray(double* A, double fillWith, int size) {
    for (int i = 0; i < size; i++) {
        A[i] = fillWith;        
    }
}

void fill2Darray(double** A, double fillWith, int row, int col) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {

            A[i][j] = fillWith;
        }
    }
}

void fill2DarrayInt(int** A, int fillWith, int row, int col) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {

            A[i][j] = fillWith;
        }
    }
}

void print2D_arr_double(double** D, int lenRow, int lenCol) {
	std::cout << "[";
	for (int i = 0; i < lenRow; i++) {
		std::cout << "[";
		for (int j = 0; j < lenCol; j++) {
			std::cout << D[i][j] << " ";
		}
		std::cout << "]";
		if (i != lenRow - 1)
			std::cout << endl;
	}
	std::cout << "] " << endl;
}

bool testEveryThing(double** Q, double* C, int* X, int* List, int lenList, int lenX) {
    bool Flag = true;
    double E = energy(Q, C, X, lenX);
    double Ed = E;
    double* H = new double[lenX];
    computeH(Q, C, H, X, lenX);
    printH(H, lenX, "H");
    double** Del_H = Declare2D_Array(lenX, lenX);
    delta_H(Q, Del_H, X, lenX);
    print2D_arr_double(Del_H, lenX, lenX);
    string Logstr = "";
    int j;
    for (int i = 0; i < lenList; i++) {
        j = List[i];
        Logstr = "bit ";
        Logstr = Logstr + to_string(j) + " : " + to_string(X[j]) + " -> " + to_string(1 - X[j]);
        double Del_E = deltaEnergyQUBO(Q, X, j, H);
        X[j] = 1 - X[j];
        Logstr = Logstr + "/\t E: " + to_string(E) + " -> ";
        E = energy(Q, C, X, lenX);
        Logstr = Logstr + to_string(E);
        Ed = Energy_based_Delta(Ed, Del_E);
        Logstr = Logstr + " Ed: " + to_string(Ed);
        if (E != Ed) {
            Flag = false;
            cout << "ERROR " << E << " ? " << Ed << endl;
        }
        updateH(H, Del_H, lenX, j);
        //printH(H, lenX);
        update_delta_H(Del_H, j, lenX);
        cout << Logstr << endl;
    }
    return Flag;
}


void testHamiltonianPreparation(double** Q, double* C, int lenX) {
    int X[] = { 0,0,0,0,0 };
    int List[] = { 0, 0, 0, 0, 0, 4, 1, 3 };
    bool Flag = testEveryThing(Q, C, X, List, 8, lenX);
    cout << Flag << endl;

}


double** convertDelHtoGpuDelH(double*** DelH, int num_replicas, int lenY) {
    double** GpuDelH = Declare2D_Array(num_replicas, lenY*lenY);

    for (int r = 0; r < num_replicas; r++) {
        for (int i = 0; i < lenY; i++) {
            for (int j = 0; j < lenY; j++) {
                GpuDelH[r][j * lenY + i] = DelH[r][i][j];
            }
        }
    }
    return GpuDelH;
}

double* VectorizedDelH(double*** DelH, int num_replicas, int lenY) {
    double* VectorDelH = new double [num_replicas* lenY*lenY];
        

    for (int r = 0; r < num_replicas; r++) {
        for (int i = 0; i < lenY; i++) {
            for (int j = 0; j < lenY; j++) {
                VectorDelH[r*lenY* lenY + i * lenY + j] = DelH[r][i][j];
            }
        }
    }
    return VectorDelH;
}



int findMinIndex(const double* array, int size) {
    int index = -1;
    if (size <= 0) {
        // Handle invalid input
        printf("Invalid array size\n");
        return -1; // You can choose a sentinel value or use an error-handling mechanism
    }

    double min = array[0]; // Assume the first element is the maximum

    for (int i = 1; i < size; ++i) {
        if (array[i] < min) {
            min = array[i];
            index = i;
        }
    }

    return index;
}


void unVectorData(int* vector_Y, int** Y, double* vector_E, double** E, int numberOfIteration, int number_replica, int lenY ) {

    for (int i = 0; i < number_replica; i++) {
        for (int j = 0; j < numberOfIteration; j++) {
            E[i][j] = vector_E[i * numberOfIteration + j];
        }
        for (int k = 0; k < lenY; k++) {
            Y[i][k] = vector_Y[i * lenY + k];
        }
    }

}

void intitTemperature(int num_replicas, double minTemp, double maxTemp, double* Temperature) {
    if (num_replicas != 1) {
        for (int r = 0; r < num_replicas; r++) {
            Temperature[r] = minTemp + r * (maxTemp - minTemp) / (num_replicas - 1);
            cout << "Temperature: " << Temperature[r] << endl;
        }
    }
    else {
        Temperature[0] = minTemp;
        cout << "Temperature: " << Temperature[0] << endl;
    }
}