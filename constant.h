#pragma once



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
#define DEBUG_INIT_CONFIG			128
#define DEBUG_SAVE_W_MATRIX			256

