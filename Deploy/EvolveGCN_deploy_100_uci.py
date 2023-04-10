import sys
import numpy as np 
import os
import time as time_1
import math
from PIL import Image
from datetime import datetime
from pynq import Overlay
import pynq
import struct
from multiprocessing import Process, Pipe, Queue, Event, Manager

def to_fixed_point(dst, src, *, width=None, iwidth, signed=True):
    if width is None:
        width = dst.dtype.itemsize * 8

    fwidth = width - iwidth
    epsilon = 1.0 / (2.0 ** fwidth)
    min_ = -1.0 * (2.0 ** (iwidth - 1)) if signed else 0.0
    max_ = (2.0 ** (iwidth - (1 if signed else 0))) - epsilon

    src = np.copy(src)
    src = src.reshape(dst.shape)
    src[src < min_] = min_
    src[src > max_] = max_
    if signed:
        src[src < 0] += (2 ** iwidth)
    dst[:] = np.around(src * (2.0 ** fwidth)).astype(dst.dtype)

def from_fixed_point(src, *, width=None, iwidth, signed=True):
    if width is None:
        width = src.dtype.itemsize * 8

    fwidth = width - iwidth
    dst = np.array(src, dtype='<f4')
    if signed:
        dst[dst >= (2 ** (width - 1))] -= (2 ** width)
    dst /= 2.0 ** fwidth
    return dst

def deleteDuplicatedElementFromList3(listA):
        #return list(set(listA))
        return sorted(set(listA), key = listA.index)
def compare_similarity(a, b):
    a_set = set(a)
    b_set = set(b)
    set_union = a_set & b_set
    set_union_list = list(set_union)
    set_store_list = a_set - set_union
    new_node_list = b_set - set_union
    return sorted(set_union_list), sorted(set_store_list), sorted(new_node_list)


node_embedding = np.load("./uci/node_embedding.npy")

edge_embedding = np.load("./uci/edge_embedding.npy")

node_num = np.load("./uci/node_number.npy")
edge_num = np.load("./uci/edge_number.npy")
edge_info = np.load("./uci/graph_info_uci.npy")

neighborhood_ref_table = np.load("./uci/neighborhood_ref_table.npy")
neighborhood_ref_table = np.array(neighborhood_ref_table, dtype = 'int')
neighborhood_reverse_ref_table = np.load("./uci/neighborhood_reverse_ref_table.npy")

overlay = Overlay("./EvolveGCN_100.bit")

TIME_STEP = 10
SUB_GRAPH_NODE_NUM = 600
SUB_GRAPH_EDGE_NUM = 1800
NUM_NODE = 3783

WEIGHT_LAYER1_INPUTDIM = 67
WEIGHT_LAYER1_OUTPUTDIM = 122


source_node_list  = [[] for i in range(192)]
edge_num_1 = []
node_num_1 = []

for i in range(192):
    for j in range(edge_num[i]):
        source_node_list[i].append(edge_info[i][j][0])
source_node_index_list = lists = [[] for i in range(192)]
for i in range(192):
   source_node_index_list[i]  = deleteDuplicatedElementFromList3(source_node_list[i])
   
edge_list_in_cpu = pynq.allocate(shape=(TIME_STEP,SUB_GRAPH_EDGE_NUM * 2), dtype=np.int32)
node_number_time_cpu = pynq.allocate(shape=(TIME_STEP), dtype=np.int32)
edge_number_time_cpu = pynq.allocate(shape=(TIME_STEP), dtype=np.int32)
neighborhood_ref_table_time_cpu = pynq.allocate(shape=(TIME_STEP, NUM_NODE), dtype=np.int32)
neighborhood_reverse_ref_table_time_cpu = pynq.allocate(shape=(TIME_STEP, SUB_GRAPH_NODE_NUM), dtype=np.int32)


gnn_layer_1_weight_cpu = pynq.allocate(shape=(WEIGHT_LAYER1_INPUTDIM, WEIGHT_LAYER1_OUTPUTDIM), dtype=np.float32)

rnn_update_weight_W_cpu = pynq.allocate(shape=(WEIGHT_LAYER1_INPUTDIM, WEIGHT_LAYER1_INPUTDIM), dtype=np.float32)
rnn_update_weight_U_cpu = pynq.allocate(shape=(WEIGHT_LAYER1_INPUTDIM, WEIGHT_LAYER1_INPUTDIM), dtype=np.float32)
rnn_update_bias_cpu = pynq.allocate(shape=(WEIGHT_LAYER1_INPUTDIM, WEIGHT_LAYER1_OUTPUTDIM), dtype=np.float32)

rnn_reset_weight_W_cpu = pynq.allocate(shape=(WEIGHT_LAYER1_INPUTDIM, WEIGHT_LAYER1_INPUTDIM), dtype=np.float32)
rnn_reset_weight_U_cpu = pynq.allocate(shape=(WEIGHT_LAYER1_INPUTDIM, WEIGHT_LAYER1_INPUTDIM), dtype=np.float32)
rnn_reset_bias_cpu = pynq.allocate(shape=(WEIGHT_LAYER1_INPUTDIM, WEIGHT_LAYER1_OUTPUTDIM), dtype=np.float32)

rnn_htilda_weight_W_cpu = pynq.allocate(shape=(WEIGHT_LAYER1_INPUTDIM, WEIGHT_LAYER1_INPUTDIM), dtype=np.float32)
rnn_htilda_weight_U_cpu = pynq.allocate(shape=(WEIGHT_LAYER1_INPUTDIM, WEIGHT_LAYER1_INPUTDIM), dtype=np.float32)
rnn_htilda_bias_cpu = pynq.allocate(shape=(WEIGHT_LAYER1_INPUTDIM, WEIGHT_LAYER1_OUTPUTDIM), dtype=np.float32)

node_embedding_cpu = pynq.allocate(shape=(TIME_STEP,SUB_GRAPH_NODE_NUM, WEIGHT_LAYER1_INPUTDIM), dtype=np.float32)
edge_embedding_cpu = pynq.allocate(shape=(TIME_STEP,SUB_GRAPH_EDGE_NUM, WEIGHT_LAYER1_INPUTDIM), dtype=np.float32)

output_cpu = pynq.allocate(shape=(SUB_GRAPH_NODE_NUM, WEIGHT_LAYER1_OUTPUTDIM), dtype=np.float32)

gnn_layer_1_weight_cpu = np.load("./DGNN_parameters/EvolveGCN_weights/gnn_weight.npy")


rnn_update_weight_W_cpu = np.load("./DGNN_parameters/EvolveGCN_weights/rnn_update_weight_W.npy")
rnn_update_weight_U_cpu = np.load("./DGNN_parameters/EvolveGCN_weights/rnn_update_weight_U.npy")
rnn_update_weight_bias_cpu = np.load("./DGNN_parameters/EvolveGCN_weights/rnn_update_weight_bias.npy")


rnn_reset_weight_W_cpu = np.load("./DGNN_parameters/EvolveGCN_weights/rnn_reset_weight_W.npy")
rnn_reset_weight_U_cpu = np.load("./DGNN_parameters/EvolveGCN_weights/rnn_reset_weight_U.npy")
rnn_reset_weight_bias_cpu = np.load("./DGNN_parameters/EvolveGCN_weights/rnn_reset_weight_bias.npy")


rnn_htilda_weight_W_cpu = np.load("./DGNN_parameters/EvolveGCN_weights/rnn_htilda_weight_W.npy")
rnn_htilda_weight_U_cpu = np.load("./DGNN_parameters/EvolveGCN_weights/rnn_htilda_weight_U.npy")
rnn_htilda_weight_bias_cpu = np.load("./DGNN_parameters/EvolveGCN_weights/rnn_htilda_weight_bias.npy") 

gnn_layer_1_weight_fixed = pynq.allocate(shape=(WEIGHT_LAYER1_INPUTDIM, WEIGHT_LAYER1_OUTPUTDIM), dtype='u4')

rnn_update_weight_W_fixed = pynq.allocate(shape=(WEIGHT_LAYER1_INPUTDIM, WEIGHT_LAYER1_INPUTDIM), dtype='u4')
rnn_update_weight_U_fixed = pynq.allocate(shape=(WEIGHT_LAYER1_INPUTDIM, WEIGHT_LAYER1_INPUTDIM), dtype='u4')
rnn_update_bias_fixed = pynq.allocate(shape=(WEIGHT_LAYER1_INPUTDIM, WEIGHT_LAYER1_OUTPUTDIM), dtype='u4')

rnn_reset_weight_W_fixed = pynq.allocate(shape=(WEIGHT_LAYER1_INPUTDIM, WEIGHT_LAYER1_INPUTDIM), dtype='u4')
rnn_reset_weight_U_fixed = pynq.allocate(shape=(WEIGHT_LAYER1_INPUTDIM, WEIGHT_LAYER1_INPUTDIM), dtype='u4')
rnn_reset_bias_fixed = pynq.allocate(shape=(WEIGHT_LAYER1_INPUTDIM, WEIGHT_LAYER1_OUTPUTDIM), dtype='u4')

rnn_htilda_weight_W_fixed = pynq.allocate(shape=(WEIGHT_LAYER1_INPUTDIM, WEIGHT_LAYER1_INPUTDIM), dtype='u4')
rnn_htilda_weight_U_fixed = pynq.allocate(shape=(WEIGHT_LAYER1_INPUTDIM, WEIGHT_LAYER1_INPUTDIM), dtype='u4')
rnn_htilda_bias_fixed = pynq.allocate(shape=(WEIGHT_LAYER1_INPUTDIM, WEIGHT_LAYER1_OUTPUTDIM), dtype='u4')

node_embedding_fixed = pynq.allocate(shape=(TIME_STEP,SUB_GRAPH_NODE_NUM, WEIGHT_LAYER1_INPUTDIM), dtype='u4')
edge_embedding_fixed = pynq.allocate(shape=(TIME_STEP,SUB_GRAPH_EDGE_NUM, WEIGHT_LAYER1_INPUTDIM), dtype='u4')

output_fixed = pynq.allocate(shape=(SUB_GRAPH_NODE_NUM, WEIGHT_LAYER1_OUTPUTDIM), dtype='u4')

to_fixed_point(gnn_layer_1_weight_fixed, gnn_layer_1_weight_cpu, width=32, iwidth=10)
to_fixed_point(rnn_update_weight_W_fixed, rnn_update_weight_W_cpu, width=32, iwidth=10)
to_fixed_point(rnn_update_weight_U_fixed, rnn_update_weight_U_cpu, width=32, iwidth=10)
to_fixed_point(rnn_update_bias_fixed, rnn_update_weight_bias_cpu, width=32, iwidth=10)
to_fixed_point(rnn_reset_weight_W_fixed, rnn_reset_weight_W_cpu, width=32, iwidth=10)
to_fixed_point(rnn_reset_weight_U_fixed, rnn_reset_weight_U_cpu, width=32, iwidth=10)
to_fixed_point(rnn_reset_bias_fixed, rnn_reset_weight_bias_cpu, width=32, iwidth=10)
to_fixed_point(rnn_htilda_weight_W_fixed, rnn_htilda_weight_W_cpu, width=32, iwidth=10)
to_fixed_point(rnn_htilda_weight_U_fixed, rnn_htilda_weight_U_cpu, width=32, iwidth=10)
to_fixed_point(rnn_htilda_bias_fixed, rnn_htilda_weight_bias_cpu, width=32, iwidth=10)

DGNN = overlay.GCN_RNN_compute_one_0

DGNN.write(0x4c,gnn_layer_1_weight_fixed.physical_address)
DGNN.write(0x58,rnn_update_weight_W_fixed.physical_address)
DGNN.write(0x64,rnn_update_weight_U_fixed.physical_address)
DGNN.write(0x70,rnn_update_bias_fixed.physical_address)
DGNN.write(0x7c,rnn_reset_weight_W_fixed.physical_address)
DGNN.write(0x88,rnn_reset_weight_U_fixed.physical_address)
DGNN.write(0x94,rnn_reset_bias_fixed.physical_address)
DGNN.write(0xa0,rnn_htilda_weight_W_fixed.physical_address)
DGNN.write(0xac,rnn_htilda_weight_U_fixed.physical_address)
DGNN.write(0xb8,rnn_htilda_bias_fixed.physical_address)

step = pynq.allocate(shape=(100), dtype=np.int32)

time_step = 2
step[0] = 2

edge_total = 0
time_total = 0.0
for step_1 in range(1):
    edge_total_internel = 0
    for time in range(time_step):
        for nd in range (node_num[step_1 + time]):
            node_embedding_cpu[time][nd] = node_embedding[source_node_index_list[step_1 + time][nd]]
        for e in range (edge_num[step_1 + time]):
            edge_embedding_cpu[time][e] =  edge_embedding[edge_total + edge_total_internel + e]
        edge_total_internel = edge_total_internel + edge_num[step_1 + time]
    edge_total = edge_total + edge_num[step_1]
    for time in range (time_step):
        node_number_time_cpu[time] = node_num[step_1 + time]
        edge_number_time_cpu[time] = edge_num[step_1 + time]
        neighborhood_ref_table_time_cpu[time] = neighborhood_ref_table[step_1 + time]
        for nd in range (node_num[step_1 + time]):
            neighborhood_reverse_ref_table_time_cpu[time][nd] = neighborhood_reverse_ref_table[step_1 + time][nd]
        #np.copyto(neighborhood_reverse_ref_table_time_cpu[time][0:SUB_GRAPH_NODE_NUM], neighborhood_reverse_ref_table_temp[step_1 + time][0:SUB_GRAPH_NODE_NUM])
    for time in range (time_step):
        for e in range (edge_num[step_1 + time]):
            edge_list_in_cpu[time][2 * e] = edge_info[step_1 + time][e][0]
            edge_list_in_cpu[time][2 * e + 1] = edge_info[step_1 + time][e][1]
            
    to_fixed_point(node_embedding_fixed, node_embedding_cpu, width=32, iwidth=10)
    to_fixed_point(edge_embedding_fixed, edge_embedding_cpu, width=32, iwidth=10)
    
    
    DGNN.write(0x10,edge_list_in_cpu.physical_address)
    DGNN.write(0x1c,node_number_time_cpu.physical_address)
    DGNN.write(0x28,edge_number_time_cpu.physical_address)
    DGNN.write(0x34,neighborhood_ref_table_time_cpu.physical_address)
    DGNN.write(0x40,neighborhood_reverse_ref_table_time_cpu.physical_address) 

    DGNN.write(0xc4,node_embedding_fixed.physical_address)
    DGNN.write(0xd0,edge_embedding_fixed.physical_address)
    DGNN.write(0xdc,output_fixed.physical_address)
    DGNN.write(0xe8,step.physical_address)
    
    start = time_1.time()
    DGNN.write(0x00,1)
    isready = DGNN.read(0x00)
    #start = time_1.time()
    while( isready == 1 ):
        isready = DGNN.read(0x00)
    end = time_1.time()
    total_time = end - start
    time_total = time_total + total_time
print('Total time: ' + str(time_total) + ' s')
   