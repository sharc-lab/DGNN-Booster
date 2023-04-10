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

#node_embedding = np.load("/home/ubuntu/DGNN/DGNN_parameters/node_embedding_wiki/node_embedding.npy")

edge_embedding = np.load("./DGNN_parameters/edge_embedding_wiki/edge_embedding.npy")

#node_num = np.load("/home/ubuntu/DGNN/DGNN_parameters/node_num/node_num_wiki.npy")
#edge_num = np.load("/home/ubuntu/DGNN/DGNN_parameters/edge_num/edge_num_wiki.npy")
edge_info = np.load("./DGNN_parameters/edge_info/edge_info.npy")
edge_number_time = np.load("./bitcoin/edge_num.npy")
neighborhood_ref_table = np.load("./DGNN_parameters/renumbering_table/neighborhood_ref_table.npy")
neighborhood_ref_table = np.array(neighborhood_ref_table, dtype = 'int')
neighborhood_reverse_ref_table = np.load("./DGNN_parameters/renumbering_table/neighborhood_reverse_ref_table.npy")
neighborhood_reverse_ref_table = np.array(neighborhood_reverse_ref_table, dtype = 'int')

overlay = Overlay("./convLSTM_100.bit")

SUB_GRAPH_NODE_NUM = 600
SUB_GRAPH_EDGE_NUM = 1800
NUM_NODE = 3783
INPUTDIM = 67
OUTPUTDIM = 67

source_node_list  = [[] for i in range(137)]
edge_num_1 = []
node_num_1 = []

for i in range(137):
    for j in range(edge_number_time[i]):
        source_node_list[i].append(edge_info[i][j][0])
source_node_index_list = lists = [[] for i in range(137)]
for i in range(137):
   source_node_index_list[i]  = deleteDuplicatedElementFromList3(source_node_list[i])
   
edge_list_in_cpu = pynq.allocate(shape=(SUB_GRAPH_EDGE_NUM * 2), dtype=np.int32)
node_number_time_cpu = pynq.allocate(shape=(192), dtype=np.int32)
edge_number_time_cpu = pynq.allocate(shape=(192), dtype=np.int32)
neighborhood_ref_table_time_cpu = pynq.allocate(shape=(NUM_NODE), dtype=np.int32)
neighborhood_reverse_ref_table_time_cpu = pynq.allocate(shape=(578), dtype=np.int32)

gnn_update_weight_W_cpu = pynq.allocate(shape=(INPUTDIM, OUTPUTDIM), dtype=np.float32)
gnn_update_weight_U_cpu = pynq.allocate(shape=(INPUTDIM, OUTPUTDIM), dtype=np.float32)
gnn_update_bias_cpu = pynq.allocate(shape=(OUTPUTDIM), dtype=np.float32)

gnn_reset_weight_W_cpu = pynq.allocate(shape=(INPUTDIM, OUTPUTDIM), dtype=np.float32)
gnn_reset_weight_U_cpu = pynq.allocate(shape=(INPUTDIM, OUTPUTDIM), dtype=np.float32)
gnn_reset_bias_cpu = pynq.allocate(shape=(OUTPUTDIM), dtype=np.float32)

gnn_htilda_weight_W_cpu = pynq.allocate(shape=(INPUTDIM, OUTPUTDIM), dtype=np.float32)
gnn_htilda_weight_U_cpu = pynq.allocate(shape=(INPUTDIM, OUTPUTDIM), dtype=np.float32)
gnn_htilda_bias_cpu = pynq.allocate(shape=(OUTPUTDIM), dtype=np.float32)

gnn_output_weight_W_cpu = pynq.allocate(shape=(INPUTDIM, OUTPUTDIM), dtype=np.float32)
gnn_output_weight_U_cpu = pynq.allocate(shape=(INPUTDIM, OUTPUTDIM), dtype=np.float32)
gnn_output_bias_cpu = pynq.allocate(shape=(OUTPUTDIM), dtype=np.float32)

gnn_update_weight_W_cpu = np.load("./DGNN_parameters/weights/gnn_update_weight_W.npy")
gnn_update_weight_U_cpu = np.load("./DGNN_parameters/weights/gnn_update_weight_U.npy")
gnn_update_bias_cpu = np.load("./DGNN_parameters/weights/gnn_update_weight_bias.npy")

gnn_reset_weight_W_cpu = np.load("./DGNN_parameters/weights/gnn_reset_weight_W.npy")
gnn_reset_weight_U_cpu = np.load("./DGNN_parameters/weights/gnn_reset_weight_U.npy")
gnn_reset_bias_cpu = np.load("./DGNN_parameters/weights/gnn_reset_weight_bias.npy")

gnn_htilda_weight_W_cpu = np.load("./DGNN_parameters/weights/gnn_htilda_weight_W.npy")
gnn_htilda_weight_U_cpu = np.load("./DGNN_parameters/weights/gnn_htilda_weight_U.npy")
gnn_htilda_bias_cpu = np.load("./DGNN_parameters/weights/gnn_htilda_weight_bias.npy")

gnn_output_weight_W_cpu = np.load("./DGNN_parameters/weights/gnn_output_weight_W.npy")
gnn_output_weight_U_cpu = np.load("./DGNN_parameters/weights/gnn_output_weight_U.npy")
gnn_output_bias_cpu = np.load("./DGNN_parameters/weights/gnn_output_weight_bias.npy")

gnn_update_weight_W_fixed = pynq.allocate(shape=(INPUTDIM, OUTPUTDIM), dtype='u4')
gnn_update_weight_U_fixed = pynq.allocate(shape=(INPUTDIM, OUTPUTDIM), dtype='u4')
gnn_update_bias_fixed = pynq.allocate(shape=(OUTPUTDIM), dtype='u4')

gnn_reset_weight_W_fixed = pynq.allocate(shape=(INPUTDIM, OUTPUTDIM), dtype='u4')
gnn_reset_weight_U_fixed = pynq.allocate(shape=(INPUTDIM, OUTPUTDIM), dtype='u4')
gnn_reset_bias_fixed = pynq.allocate(shape=(OUTPUTDIM), dtype='u4')

gnn_htilda_weight_W_fixed = pynq.allocate(shape=(INPUTDIM, OUTPUTDIM), dtype='u4')
gnn_htilda_weight_U_fixed = pynq.allocate(shape=(INPUTDIM, OUTPUTDIM), dtype='u4')
gnn_htilda_bias_fixed = pynq.allocate(shape=(OUTPUTDIM), dtype='u4')

gnn_output_weight_W_fixed = pynq.allocate(shape=(INPUTDIM, OUTPUTDIM), dtype='u4')
gnn_output_weight_U_fixed = pynq.allocate(shape=(INPUTDIM, OUTPUTDIM), dtype='u4')
gnn_output_bias_fixed = pynq.allocate(shape=(OUTPUTDIM), dtype='u4')

cell_state_cpu = pynq.allocate(shape=(NUM_NODE,INPUTDIM), dtype=np.float32)
node_embedding_cpu = pynq.allocate(shape=(NUM_NODE,INPUTDIM), dtype=np.float32)
edge_embedding_cpu = pynq.allocate(shape=(SUB_GRAPH_EDGE_NUM, INPUTDIM), dtype=np.float32)

cell_state_fixed = pynq.allocate(shape=(NUM_NODE,INPUTDIM), dtype='u4')
node_embedding_fixed = pynq.allocate(shape=(NUM_NODE,INPUTDIM), dtype='u4')
edge_embedding_fixed = pynq.allocate(shape=(SUB_GRAPH_EDGE_NUM, INPUTDIM), dtype='u4')

node_embedding_cpu = np.load("./DGNN_parameters/node_embedding_wiki/node_embedding.npy")
node_number_time = np.load("./bitcoin/node_num.npy")
edge_number_time = np.load("./bitcoin/edge_num.npy")

np.copyto(node_number_time_cpu,node_number_time)
np.copyto(edge_number_time_cpu, edge_number_time)

to_fixed_point(gnn_update_weight_W_fixed, gnn_update_weight_W_cpu, width=32, iwidth=10)
to_fixed_point(gnn_update_weight_U_fixed, gnn_update_weight_U_cpu, width=32, iwidth=10)
to_fixed_point(gnn_update_bias_fixed, gnn_update_bias_cpu, width=32, iwidth=10)

to_fixed_point(gnn_reset_weight_W_fixed, gnn_reset_weight_W_cpu, width=32, iwidth=10)
to_fixed_point(gnn_reset_weight_U_fixed, gnn_reset_weight_U_cpu, width=32, iwidth=10)
to_fixed_point(gnn_reset_bias_fixed, gnn_reset_bias_cpu, width=32, iwidth=10)

to_fixed_point(gnn_htilda_weight_W_fixed, gnn_htilda_weight_W_cpu, width=32, iwidth=10)
to_fixed_point(gnn_htilda_weight_U_fixed, gnn_htilda_weight_U_cpu, width=32, iwidth=10)
to_fixed_point(gnn_htilda_bias_fixed, gnn_htilda_bias_cpu, width=32, iwidth=10)

to_fixed_point(gnn_output_weight_W_fixed, gnn_output_weight_W_cpu, width=32, iwidth=10)
to_fixed_point(gnn_output_weight_U_fixed, gnn_output_weight_U_cpu, width=32, iwidth=10)
to_fixed_point(gnn_output_bias_fixed, gnn_output_bias_cpu, width=32, iwidth=10)

to_fixed_point(node_embedding_fixed, node_embedding_cpu, width=32, iwidth=10)
to_fixed_point(cell_state_fixed, cell_state_cpu, width=32, iwidth=10)

DGNN = overlay.GCN_RNN_compute_one_0

DGNN.write(0x1c,node_number_time_cpu.physical_address)
DGNN.write(0x28,edge_number_time_cpu.physical_address)

DGNN.write(0x34,gnn_update_weight_W_fixed.physical_address)
DGNN.write(0x40,gnn_update_weight_U_fixed.physical_address)
DGNN.write(0x4c,gnn_update_bias_fixed.physical_address)

DGNN.write(0x58,gnn_reset_weight_W_fixed.physical_address)
DGNN.write(0x64,gnn_reset_weight_U_fixed.physical_address)
DGNN.write(0x70,gnn_reset_bias_fixed.physical_address)

DGNN.write(0x7c,gnn_htilda_weight_W_fixed.physical_address)
DGNN.write(0x88,gnn_htilda_weight_U_fixed.physical_address)
DGNN.write(0x94,gnn_htilda_bias_fixed.physical_address)

DGNN.write(0xa0,gnn_output_weight_W_fixed.physical_address)
DGNN.write(0xac,gnn_output_weight_U_fixed.physical_address)
DGNN.write(0xb8,gnn_output_bias_fixed.physical_address)

DGNN.write(0xc4,node_embedding_fixed.physical_address)
DGNN.write(0xd0,cell_state_fixed.physical_address)

time = pynq.allocate(shape=(100), dtype=np.int32)
edge_total = 0

time_total = 0

for time_step in range(137):
    time[0] = time_step
    #neighborhood_ref_table_time_cpu = neighborhood_ref_table[time_step]
    #neighborhood_reverse_ref_table_time_cpu = neighborhood_reverse_ref_table[time_step]
    np.copyto(neighborhood_ref_table_time_cpu,neighborhood_ref_table[time_step])
    np.copyto(neighborhood_reverse_ref_table_time_cpu, neighborhood_reverse_ref_table[time_step])
    
    for e in range (edge_number_time[time_step]):
            edge_list_in_cpu[2 * e] = edge_info[time_step][e][0]
            edge_list_in_cpu[2 * e + 1] = edge_info[time_step][e][1]
    for e in range (edge_number_time[time_step]):
            edge_embedding_cpu[e] =  edge_embedding[edge_total + e]
    edge_total = edge_total + edge_number_time[time_step]
    to_fixed_point(edge_embedding_fixed, edge_embedding_cpu, width=32, iwidth=10)
    DGNN.write(0x10,edge_list_in_cpu.physical_address)
    DGNN.write(0xdc,edge_embedding_fixed.physical_address)
    DGNN.write(0xe8,neighborhood_ref_table_time_cpu.physical_address)
    DGNN.write(0xf4,neighborhood_reverse_ref_table_time_cpu.physical_address)
    DGNN.write(0x100,time.physical_address)
    
    
    DGNN.write(0x00,1)
    isready = DGNN.read(0x00)
    start = time_1.time()
    while( isready == 1 ):
        isready = DGNN.read(0x00)
    end = time_1.time()
    total_time = end - start
    time_total = time_total + total_time
print('Total time: ' + str(time_total) + ' s')

