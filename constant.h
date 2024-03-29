#pragma once


#define TEMPERATURE_GEOMETRIC	1


//#define NumberOfIteration       10000
#define ReadDataFromFile        3    
#define GenerateDataRandomly    2
#define FillWithOne             1
#define FillWithMinusOne        -1
#define FillWithZero            0
#define IsingMode               -1
#define QUBOMode                0
#define QUBOGPU                 1
#define QUBOGPUFULL				2
#define DEBUGFLAG				false

#define NORMAL					0
#define NONSYMETRIC				1
#define UPPER_TRIANGULAR		2
#define WITH_BIAS				4


/* 
Debug print format:
step, blockId, threadId, action, Energy, deltaE
*/
#define NO_DEBUG					0
#define DEBUG_RANDOM_FLIP			1			// Debug for finding if there is any fliping bit suggestion due to random value (Metropolis)
#define DEBUG_SELECTED_FLIP			2
#define DEBUG_FIND_BEST_ENERGY		4
#define DEBUG_UPDATE_H				8
#define DEBUG_EXCHANGE				16
#define DEBUG_SAVE_DEVICE_RESULT	32
#define DEBUG_DELTA_FLIP			64
#define ALL_DEBUG_ADMIN				127
#define DEBUG_INIT_CONFIG			128
#define DEBUG_SAVE_W_MATRIX			256
#define DEBUG_WARNNING_AND_INFO		512
#define DEBUG_FINAL_CONFIG			1024
#define DEBUG_ENERGY_RECORD_LOG		2048
#define DEBUG_MAGNET_RECORD_LOG		4096


