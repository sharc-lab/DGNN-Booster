#include <cstddef>
//#include "D:\vivado2018setup\Vivado\2018.3\include\gmp.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <ap_fixed.h>
#include "hls_stream.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>

#include <algorithm>

typedef ap_fixed<32, 10> FM_TYPE;
typedef ap_fixed<32, 10> WT_TYPE;

#define MAX_EDGE 500
#define MAX_NODE 200
#define MAX_DEGREE 20

#define LAYER_NUM 1
#define EMB_DIM 100
#define NUM_TASK 1
#define MLP_1_IN 100
#define MLP_1_OUT 200
#define MLP_2_IN 200
#define MLP_2_OUT 100
#define MLP_IN_MAX 200
#define MLP_OUT_MAX 200

#define E_EPS 0.00001

#define ND_FEATURE 9
#define EDGE_ATTR 3



