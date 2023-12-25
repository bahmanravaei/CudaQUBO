#pragma once


#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <filesystem>
#include <sys/stat.h>
#include "helpFunction.h"
#include "in_out_functions.h"
#include "constant.h"

using namespace std;


/*  Return an       */
// Function to create a 2D array of integers to represent spin vectors for Ising or QUBO system
// Parameters:
// - ExecuteMode: Integer value representing the mode (Ising or QUBO)
// - lenY: Length of the spin vector
// - num_replicas: Number of replicas
// Return the pointer to the 2D array of integers representing the spin vectors for number of replicas in replica exchange MCMC

int** createVector(int ExecuteMode, int lenY, int num_replicas) {
    int** Y = new int* [num_replicas];
    for (int r = 0; r < num_replicas; r++) {
        // Initialize a new spin vector for each replica
        Y[r] = new int[lenY];
        for (int i = 0; i < lenY; i++) {
            Y[r][i] = rand() % 2;

            // If the ExecuteMode is Ising, convert 0 to -1
            if (ExecuteMode == IsingMode && Y[r][i] == 0)
                Y[r][i] = Y[r][i] - 1;
        }
    }
    return Y;
}



/* Initialize the bias based on Flag:
*               FillWithZero -> set zero,
*               FillWithOne -> set 1 (One),
*               FillWithMinusOne -> set -1 (Negative One)
*               GenerateDataRandomly -> set randomly
*               ReadDataFromFile -> Read B from file Bfile
*/
double* initB(int len, int Flag, string Bfile) {
    int minB = -10; int maxB = 10;
    double* B = new double[len];

    if (Flag == ReadDataFromFile) {
        if (Bfile == "") {
            Flag = FillWithZero;
            cout << "B filled with zero\n";
        }
        else {
            ReadVector(B, Bfile);
            return B;
        }
    }

    for (int i = 0; i < len; i++) {
        if (Flag == FillWithZero)
            B[i] = 0;
        else if (Flag == FillWithOne)
            B[i] = 1;
        else if (Flag == FillWithMinusOne)
            B[i] = -1;
        else if (Flag == GenerateDataRandomly)
            B[i] = rand() % (maxB - minB) + minB;
    }
    if (Flag == ReadDataFromFile)
        ReadVector(B, Bfile);
    return B;
}



/* Initialize the matrix of interconnection between spins
*   input:
*       lenX:   number of spins
*       lenMat: lenght of lattice for Ising Model
*       Flag:   How to init the interconnection matrix,
*           if Flag == ReadDataFromFile -> read the interconnection from "Wfile" file
*           otherwise -> consider model as 2D lattice, and init the interconneciton
*   Output:
*       return the matrix of interconnection
*/
double** initW(int lenX, int lenMat, int Flag, string Wfile) {
    double** W = new double* [lenX];

    //Create the Matrix W
    for (int i = 0; i < lenX; i++) {
        W[i] = new double[lenX];
        for (int j = 0; j < lenX; j++) {
            W[i][j] = 0.0;
        }
    }

    if (Flag == ReadDataFromFile) {
        ReadWFromFile(W, Wfile);
        return W;
    }

    for (int i = 0; i < lenX; i++) {
        //cout <<i <<" : " << ((i - 1) % lenX) << "\t" << (i + 1) % lenX << "\t" << (i - lenMat) % lenX << "\t" << (i + lenMat) % lenX << endl;

        if (i % lenMat == 0) {
            W[i][(i + lenMat - 1)] = 1.0;
            //cout << (i + lenMat - 1) << endl;
        }
        else
            W[i][(i - 1) % lenX] = 1.0;

        if (i % lenMat == lenMat - 1) {
            W[i][i - lenMat + 1] = 1.0;
            //cout << i - lenMat + 1 << endl;
        }
        else
            W[i][(i + 1) % lenX] = 1.0;

        if (i / lenMat == 0) {
            W[i][i + lenX - lenMat] = 1.0;
            //cout << i + lenX-lenMat << endl;
        }
        else
            W[i][(i - lenMat) % lenX] = 1.0;

        W[i][(i + lenMat) % lenX] = 1.0;

    }
    return W;
}



// Calculate the energy of spin state X based on interconnection W and bias B
double energy(double** W, double* B, int* X, int lenX) {
    double E = 0;
    for (int i = 0; i < lenX; i++) {
        for (int j = 0; j < lenX; j++) {
            E += W[i][j] * X[i] * X[j];
        }
        E = E + B[i] * X[i];
    }
    //cout<<"Energy: "<<-E<<endl;
    return -E;
}

double Energy_based_Delta(double E, double delta_e) {
    return E + delta_e;
}

//  Calculate the Magnetization of spin state X
double magnetization(int* X, int lenX) {
    int M = 0;
    for (int i = 0; i < lenX; i++) {
        M += X[i];
    }
    return M;
}

// Calculate the Hamiltonian 
double calculateIsingH(double** A, double* B, int* Y, int i, int lenY) {
    double H = 0.0;
    for (int j = 0; j < lenY; j++) {
        //if(i!=j)
        H = H + A[i][j] * Y[j];
    }

    //cout << "H value: " << H << " : " << H + B[i] << endl;
    H = 2 * H;
    H = H + B[i];

    return H;
}


double calculateH(double** Q, double* C, int* X, int i, int lenX) {
    double H = 0.0;
    for (int j = 0; j < lenX; j++) {
        if (i != j) {
            H = H + Q[i][j] * X[j];
        }
    }
    H = 2 * H + Q[i][i];
    H = H + C[i];
    return H;
}



// Compute the value of Hamiltonian for all bits
double* computeH(double** Q, double* C, double* H_List, int* X, int lenX) {
    //double* H_List = new double [lenX];

    for (int i = 0; i < lenX; i++) {
        H_List[i] = calculateH(Q, C, X, i, lenX);
    }
    return H_List;

}

// dynamically allocate memory for 2d arrays
double** Declare2D_Array(int row, int col) {
    double** Matrix = new double* [row];
    for (int i = 0; i < row; i++) {
        Matrix[i] = new double[col];
    }
    return Matrix;
}

double*** Declare3D_Array(int lenDim1, int lenDim2, int lenDim3) {
    double*** ARR = new double** [lenDim1];
    for (int i = 0; i < lenDim1; i++) {
        ARR[i] = Declare2D_Array(lenDim2, lenDim3);
    }
    return ARR;
}

double** ComputeH_forAllReplica(int num_replicas, double** Q, double* C, int** X, int lenX) {
    double** H = Declare2D_Array(num_replicas, lenX);
    for (int i = 0; i < num_replicas; i++) {
        computeH(Q, C, H[i], X[i], lenX);
    }
    return H;
}

// Update value for all hamiltonian
double* updateH(double* H, double** Del_H, int lenX, int j) {
    for (int i = 0; i < lenX; i++) {
        H[i] = H[i] + Del_H[i][j];
    }
    return H;
}



// Calculate the value of delta H
double** delta_H(double** Q, double** Del_H, int* X, int lenX) {
    //double** Del_H = new double* [lenX];
    //for (int i = 0; i < lenX; i++) {
    //    Del_H[i] = new double [lenX];
    //}

    for (int i = 0; i < lenX; i++) {
        for (int j = 0; j < lenX; j++) {
            if (i == j) {
                Del_H[i][j] = 0;
            }
            else {
                Del_H[i][j] = 2 * (1 - 2 * X[j]) * Q[i][j];
            }
        }
    }
    return Del_H;
}

double*** ComputeDelH_forAllReplica(int num_replicas, double** Q, int** X, int lenX) {
    double*** Del_H = Declare3D_Array(num_replicas, lenX, lenX);
    for (int r = 0; r < num_replicas; r++) {
        delta_H(Q, Del_H[r], X[r], lenX);
    }
    return Del_H;
}
double** update_delta_H(double** Del_H, int j, int lenX) {
    for (int i = 0; i < lenX; i++) {
        Del_H[i][j] = -1 * Del_H[i][j];
    }
    return Del_H;
}


// Calculate the delta energy of spin state X if spin index i flip
// this function is based on hamiltonian (to be used in GPU model latter)
double deltaEnergyQUBO(double** Q, int* X, int i, double* H) {
    return -1 * (1 - 2 * X[i]) * H[i];

}


/* Calculate the delta energy of spin state Y if spin index i flip, (based on interconnection A and bias B)
* flip a spin ::: if ExecuteMode == Ising => -1 -> 1, 1 -> -1
* flip a spin ::: if ExecuteMode == QUBO  => 0  -> 1, 1 ->  0
*/
double deltaEnergyIsing(int ExecuteMode, double** A, double* B, int* Y, int lenY, int i) {
    double deltaE = 0;
    double H_i;
    if (ExecuteMode == IsingMode) {
        H_i = calculateIsingH(A, B, Y, i, lenY);
        deltaE = (2 * Y[i]) * H_i;
    }
    else if (ExecuteMode == QUBOMode) {
        H_i = calculateH(A, B, Y, i, lenY);
        deltaE = -1 * (1 - 2 * Y[i]) * H_i;
    }

    return deltaE;
}



void replicaExchange(double* Temperature, int num_replicas, int** Y, double** M, double** E, int step, double** H, double*** DelH, int& exchangeFlag) {

    double* delta_beta = new double[num_replicas];
    double* delta_energy = new double[num_replicas];

    for (int r = 0; r < num_replicas - 1; ++r) {
        delta_beta[r] = Temperature[r] - Temperature[r + 1];
        delta_energy[r] = (1 / Temperature[r] - 1 / Temperature[r + 1]) * (E[r][step] - E[r + 1][step]);
    }


    for (int r = 0; r < num_replicas - 1; ++r) {
        if (exchangeFlag == 0 && (delta_energy[r] > 0 || (rand() / static_cast<double>(RAND_MAX)) < exp(delta_beta[r] * delta_energy[r]))) {
            exchangeFlag = 1;
            //std::cout << "Exchange " << r << " <-> " << r + 1 << std::endl;
            std::swap(Y[r], Y[r + 1]);
            std::swap(H[r], H[r + 1]);
            std::swap(DelH[r], DelH[r + 1]);
            //spins[replica, :], spins[replica + 1, :] = spins[replica + 1, :], spins[replica, :]

            std::swap(E[r][step], E[r + 1][step]);
            std::swap(M[r][step], M[r + 1][step]);
        }
        else {
            exchangeFlag = 0;
        }
    }
}