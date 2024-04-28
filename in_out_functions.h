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
        
        //std::cout << i<<": " << element << std::endl;
    }
}



void ReadWFromFile(double** W, string Wfile, int problem_type) {
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
            if ((problem_type & NONSYMETRIC)== NONSYMETRIC) W[index1 - 1][index2 - 1] = weight;
            else { W[index2 - 1][index1 - 1] = W[index1 - 1][index2 - 1] = weight; }
                
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



void recordLogs(string outputPath, double** E, int numberOfIteration, int num_replicas, int lenX, int Lsqrt, int** X, int* bestSpinModel, int** best_config_all, int debug_mode) {

    
    if ((debug_mode& DEBUG_ENERGY_RECORD_LOG) == DEBUG_ENERGY_RECORD_LOG)
        writeListToFile(outputPath + "\\Energy.csv", E, numberOfIteration, num_replicas);
    if ((debug_mode& DEBUG_FINAL_CONFIG) == DEBUG_FINAL_CONFIG)
        writeSpinsInFile(num_replicas, X, lenX, Lsqrt, outputPath, "latticeFinal");
    if ((debug_mode & DEBUG_BEST_CONFIG_ALL_REP) == DEBUG_BEST_CONFIG_ALL_REP)
        writeSpinsInFile(num_replicas, best_config_all, lenX, Lsqrt, outputPath, "bestConfig");

    //writeSpinInFile(X[0], lenX, Lsqrt, outputPath + "\\latticeFinal.csv");

    //Mat = vectorToMatrix(bestSpinModel, Lsqrt);
    //writeMatrixToFile("latticeBest.csv", Mat, Lsqrt);
    writeSpinInFile(bestSpinModel, lenX, Lsqrt, outputPath + "\\latticeBest.csv");
}

void print_log_host(const char* action, int step, int temprature_index, int replicaIindex, int bitIndex, double previousEnergy, double deltaE, double newEnergy, double H, double del_H, int flipped_bit, int index_del_h) {
    // Use printf for printing within host functions
    //return;
    //char buffer[500];
    //sprintf(buffer, "%d, %d, %d, %d, %s, %lf, %lf, %lf, %lf, %lf, %d, %d\n", step, replicaIindex, bitIndex, action, previousEnergy, deltaE, newEnergy, H, del_H, flipped_bit, index_del_h);

    std::cout << step << "," << replicaIindex << "," << temprature_index << "," << bitIndex << "," << action << "," << previousEnergy << "," << deltaE << "," << newEnergy << "," << H << "," << del_H << "," << flipped_bit << "," << index_del_h<<endl;
}

int set_program_config(string value) {
    int program_config = 0;
    if (value == "Geometric") {
        program_config = TEMPERATURE_GEOMETRIC;
    }
    else if (value == "TEMPERATURE_CIRCULAR") {
        program_config = TEMPERATURE_CIRCULAR;
    }
    else if (value == "FAST_CONVERGE") {
        program_config = FAST_CONVERGE;
    }


    return program_config;
}

int set_problem_type(string value) {
    int problem_type = 0;
    if (value == "NONSYMETRIC") {
        problem_type = NONSYMETRIC;
    }else if (value == "UPPER_TRIANGULAR") {
        problem_type = UPPER_TRIANGULAR;
    }
    else if (value == "WITH_BIAS") {
        problem_type = WITH_BIAS;
    }
    return problem_type;
}


int setDebugMode(string value) {
    int debugMode = 0;
    if (value == "DEBUG_RANDOM_FLIP") {
        debugMode = DEBUG_RANDOM_FLIP;
    }else if (value == "DEBUG_SELECTED_FLIP") {
        debugMode = DEBUG_SELECTED_FLIP;
    }
    else if (value == "DEBUG_FIND_BEST_ENERGY") {
        debugMode = DEBUG_FIND_BEST_ENERGY;
    }
    else if (value == "DEBUG_UPDATE_H") {
        debugMode = DEBUG_UPDATE_H;
    }
    else if (value == "DEBUG_EXCHANGE") {
        debugMode = DEBUG_EXCHANGE;
    }
    else if (value == "DEBUG_SAVE_DEVICE_RESULT") {
        debugMode = DEBUG_SAVE_DEVICE_RESULT;
    }
    else if (value == "DEBUG_DELTA_FLIP") {
        debugMode = DEBUG_DELTA_FLIP;
    }
    else if (value == "DEBUG_INIT_CONFIG") {
        debugMode = DEBUG_INIT_CONFIG;
    }
    else if (value == "DEBUG_SAVE_W_MATRIX") {
        debugMode = DEBUG_SAVE_W_MATRIX;
    }
    else if (value == "DEBUG_INIT_CONFIG") {
        debugMode = DEBUG_INIT_CONFIG;
    }
    else if (value == "DEBUG_FINAL_CONFIG") {
        debugMode = DEBUG_FINAL_CONFIG;
    }
    else if (value == "DEBUG_ENERGY_RECORD_LOG") {
        debugMode = DEBUG_ENERGY_RECORD_LOG;
    }
    else if (value == "DEBUG_MAGNET_RECORD_LOG") {
        debugMode = DEBUG_MAGNET_RECORD_LOG;
    }
    else if (value == "DEBUG_BEST_CONFIG_ALL_REP") {
        debugMode = DEBUG_BEST_CONFIG_ALL_REP;
    }
    
    return debugMode;

}

/* read the configuration from the file Settings.txt and initialize the parameter  */
int readSetting(int& L, int& Lsqrt, string& Afile, string& Bfile, string& outputPath, int& ExecuteMode, int& num_replicas, int& numberOfIteration, int& exchange_attempts, int& extend_exchange_mode, int& number_of_temp, double& minTemp, double& maxTemp, int& debug_mode, int& problem_type) {
    std::ifstream fin("settings.txt");
    std::string line;
    std::string Key, Value;
    int program_config = 0;
    extend_exchange_mode = 0;
    if (fin.is_open()) {

        while (std::getline(fin, line)) {
            //cout << line << "\n";
            stringstream stringStream1(line);
            stringStream1 >> Key;
            //cout << Key << endl;

            if (Key == "SizeOfVector:")
                stringStream1 >> L;
            else if (Key == "ProblemType:") {// Description of the input QUBO problem                
                stringStream1 >> Value;
                problem_type = problem_type | set_problem_type(Value);                                                
            }
            else if (Key == "Temperature_init:") {
                stringStream1 >> Value;
                program_config = program_config | set_program_config(Value); // Description for solver configuration
                //program_config = set_program_config(Value);
            }
            else if (Key == "program_config:") {
                stringStream1 >> Value;
                program_config = program_config | set_program_config(Value); // Description for solver configuration
                //program_config = set_program_config(Value);
            }
            else if (Key == "Number_of_temp:") {
                stringStream1 >> number_of_temp;
            }
            else if (Key == "Lsqrt:")
                stringStream1 >> Lsqrt;
            //else if (Key == "InitialTemperature:")
            //    stringStream1 >> T;
            else if (Key == "InputA:")
                stringStream1 >> Afile;
            else if (Key == "InputB:") {
                stringStream1 >> Bfile;
                problem_type = problem_type | WITH_BIAS;
            }
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
            else if (Key == "debugMode:") {
                stringStream1 >> Value;
                debug_mode = debug_mode | setDebugMode(Value);
            }
        }
    }
    fin.close();
    if (number_of_temp == 0) {
        number_of_temp = num_replicas;
    }
    if (extend_exchange_mode == 0) {
        if (exchange_attempts >= 100)
            extend_exchange_mode = exchange_attempts / 10;
        else
            extend_exchange_mode = 10;
    }

    return program_config;
}
