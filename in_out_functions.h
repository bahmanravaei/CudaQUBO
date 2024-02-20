#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <filesystem>
#include <sys/stat.h>
#include "helpFunction.h"
#include "constant.h"

using namespace std;


/*
* This function takes in a file name, a 2D array named Matrix, and its size L as parameters.
* It then opens the specified file in write mode and checks if the file was successfully opened.
* If the file was opened successfully, it iterates through the Matrix and writes each element to
* the file followed by a comma  and a new line character at the end of each row.
*/
void writeMatrixToFile(string fileName, double** Matrix, int L) {
    fstream my_file;
    my_file.open(fileName, ios::out);
    if (!my_file) {
        cout << "ERROR    :   File not created!";
    }
    else {
        //cout << "test" << endl;
        for (int i = 0; i < L; i++)
        {
            for (int j = 0; j < L; j++)
            {
                my_file << Matrix[i][j];
                if (j != L - 1)
                    my_file << ',';
            }
            my_file << '\n';

        }
    }
    cout << "writeMatrixToFile before file close \n";
    my_file.close();

}

void writeIntListToFile(string fileName, int* List, int lenList) {
    fstream my_file;
    my_file.open(fileName, ios::out);
    if (!my_file) {
        cout << "ERROR    :   File not created!";
    }
    else {
        for (int i = 0; i < lenList; i++)
        {
            my_file << List[i];
            if (i != lenList - 1)
                my_file << ',';
        }

    }
    my_file.close();
    //return out;
}

void writeListToFile(string fileName, double* List, int lenList) {
    fstream my_file;
    my_file.open(fileName, ios::out);
    if (!my_file) {
        cout << "ERROR    :   File not created!";
    }
    else {
        for (int i = 0; i < lenList; i++)
        {
            my_file << List[i];
            if (i != lenList - 1)
                my_file << ',';
        }

    }
    my_file.close();
    //return out;
}


void writeListToFile(string fileName, double** List, int L, int num_replicas) {
    fstream my_file;
    my_file.open(fileName, ios::out);
    if (!my_file) {
        cout << "ERROR    :   File not created!";
    }
    else {

        for (int i = 0; i < L; i++)
        {
            for (int r = 0; r < num_replicas; r++) {
                my_file << List[r][i];
                if (r != num_replicas - 1)
                    my_file << ',';

            }
            my_file << '\n';

        }

    }
    my_file.close();
    //return out;
}




// Fill array B from  file "Bfile"
void ReadVector(double* B, string Bfile) {
    std::ifstream fin(Bfile);
    int i = 0;
    //std::vector<int> data;

    double element;
    while (fin >> element)
    {
        B[i] = element;
        i++;

        //std::cout << element << std::endl;
    }
}



void ReadWFromFile(double** W, string Wfile) {
    std::ifstream fin(Wfile);
    //std::vector<int> data;
    int node, edge;
    std::string line;
    std::string tempStr;
    int index1, index2;
    double weight;

    if (fin.is_open()) {
        std::getline(fin, line);
        stringstream stringStream1(line);
        stringStream1 >> node >> edge;
        //cout << node<< edge<< endl;
        while (std::getline(fin, line)) {
            //cout << line << "\n";
            stringstream stringStream2(line);
            stringStream2 >> index1 >> index2 >> weight;
            W[index2 - 1][index1 - 1] = W[index1 - 1][index2 - 1] = weight;
            //cout << line << endl;
            //cout << "(" << index1 << "," << index2 << ")" << W[index2-1][index1-1] << " " << W[index1-1][index2-1] << " " << weight << endl;

        }
        fin.close();
    }

}

// Function to represent the 1D array of spin in form of a 2D array (Matrix)
double** vectorToMatrix(int* X, int lenMat) {
    double** Mat = new double* [lenMat];
    for (int i = 0; i < lenMat; i++) {
        Mat[i] = new double[lenMat];
        for (int j = 0; j < lenMat; j++) {
            Mat[i][j] = X[i * lenMat + j];
        }
    }
    return Mat;
}



//write a spin state in outputfile
void writeSpinInFile(int* spin, int lenSpin, int lenMat, string outputFileName) {

    if (lenMat * lenMat == lenSpin) {
        double** Mat;
        Mat = vectorToMatrix(spin, lenMat);
        writeMatrixToFile(outputFileName, Mat, lenMat);
    }
    else {
        writeIntListToFile(outputFileName, spin, lenSpin);
    }

}


//write all spin states in outputfile
void writeSpinsInFile(int num_replicas, int** X, int lenX, int Lsqrt, string outputPath, string fileNamePrefix) {

    for (int i = 0; i < num_replicas; i++) {
        writeSpinInFile(X[i], lenX, Lsqrt, outputPath + "\\" + fileNamePrefix + to_string(i) + ".csv");
    }
}






bool folderExists(string folderName) {
    struct stat info;
    if (stat(folderName.c_str(), &info) != 0) {
        return false;
    }
    return (info.st_mode & S_IFDIR) != 0;
}


bool createFolder(string folderName) {
    if (!folderExists(folderName)) {

        wstring widestr = wstring(folderName.begin(), folderName.end());
        const wchar_t* widecstr = widestr.c_str();
        int status = _wmkdir(widecstr);
        //(folderName.c_str())
    //int status = mkdir(folderName.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        if (status == 0) {
            std::cout << "Folder created successfully" << std::endl;
            return true;
        }
        else {
            std::cout << "Failed to create folder" << std::endl;
            return false;
        }
    }
    else {
        std::cout << "Folder already exists" << std::endl;
        return false;
    }
}



void recordLogs(string outputPath, double** M, double** E, int numberOfIteration, int num_replicas, int lenX, int Lsqrt, int** X, int* bestSpinModel) {

    writeListToFile(outputPath + "\\Magnet.csv", M, numberOfIteration, num_replicas);
    writeListToFile(outputPath + "\\Energy.csv", E, numberOfIteration, num_replicas);

    writeSpinsInFile(num_replicas, X, lenX, Lsqrt, outputPath, "latticeFinal");

    //writeSpinInFile(X[0], lenX, Lsqrt, outputPath + "\\latticeFinal.csv");

    //Mat = vectorToMatrix(bestSpinModel, Lsqrt);
    //writeMatrixToFile("latticeBest.csv", Mat, Lsqrt);
    writeSpinInFile(bestSpinModel, lenX, Lsqrt, outputPath + "\\latticeBest.csv");
}



/* read the configuration from the file Settings.txt and initialize the parameter  */
void readSetting(int& L, int& Lsqrt, string& Afile, string& Bfile, string& outputPath, int& ExecuteMode, int& num_replicas, int& numberOfIteration, int& exchange_attempts, double& minTemp, double& maxTemp) {
    std::ifstream fin("setting.txt");
    std::string line;
    std::string Key, Value;

    if (fin.is_open()) {

        while (std::getline(fin, line)) {
            //cout << line << "\n";
            stringstream stringStream1(line);
            stringStream1 >> Key;
            //cout << Key << endl;

            if (Key == "SizeOfVector:")
                stringStream1 >> L;
            else if (Key == "Lsqrt:")
                stringStream1 >> Lsqrt;
            //else if (Key == "InitialTemperature:")
            //    stringStream1 >> T;
            else if (Key == "InputA:")
                stringStream1 >> Afile;
            else if (Key == "InputB:")
                stringStream1 >> Bfile;
            else if (Key == "outputDir:")
                stringStream1 >> outputPath;
            else if (Key == "num_replicas:") {
                stringStream1 >> num_replicas;
                //cout << "num_replicas: " << num_replicas << endl;
            }
            else if (Key == "NumberOfIteration:")
                stringStream1 >> numberOfIteration;
            else if (Key == "exchange_attempts:")
                stringStream1 >> exchange_attempts;
            else if (Key == "minTemp:")
                stringStream1 >> minTemp;
            else if (Key == "maxTemp:")
                stringStream1 >> maxTemp;
            else if (Key == "IsingorQUBO:") {
                stringStream1 >> Value;
                if (Value == "Ising")
                    ExecuteMode = IsingMode;
                else if (Value == "QUBO")
                    ExecuteMode = QUBOMode;
                else if (Value == "QUBOGPU")
                    ExecuteMode = QUBOGPU;
                else if (Value == "QUBOGPUFULL")
                    ExecuteMode = QUBOGPUFULL;
                else {
                    cout << "Error" << endl;
                    exit(-1);
                }
            }
        }
    }
    fin.close();

}
