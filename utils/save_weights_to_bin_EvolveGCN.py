

import torch
import numpy as np
import struct
import time as mytime

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

np.random.seed(100)
device = 'cpu'

gnn_weight = 2 *np.random.rand(67,122) - 1
gnn_weight = np.array(gnn_weight, dtype='float32')
#bb = gnn_weight

rnn_update_weight_W = 2 *np.random.rand(67,67) - 1
rnn_update_weight_U = 2 *np.random.rand(67,67) - 1
rnn_update_weight_bias = 2 *np.random.rand(67,122) - 1

rnn_update_weight_W = np.array(rnn_update_weight_W,dtype='float32' )
rnn_update_weight_U = np.array(rnn_update_weight_U,dtype='float32' )
rnn_update_weight_bias = np.array(rnn_update_weight_bias,dtype='float32' )


rnn_reset_weight_W = 2 *np.random.rand(67,67) - 1
rnn_reset_weight_U = 2 *np.random.rand(67,67) - 1
rnn_reset_weight_bias = 2 *np.random.rand(67,122) - 1

rnn_reset_weight_W = np.array(rnn_reset_weight_W,dtype='float32' )
rnn_reset_weight_U = np.array(rnn_reset_weight_U,dtype='float32' )
rnn_reset_weight_bias = np.array(rnn_reset_weight_bias,dtype='float32' )


rnn_htilda_weight_W = 2 *np.random.rand(67,67) - 1
rnn_htilda_weight_U = 2 *np.random.rand(67,67) - 1
rnn_htilda_weight_bias = 2 *np.random.rand(67,122) - 1
#aa = rnn_htilda_weight_bias

rnn_htilda_weight_W = np.array(rnn_htilda_weight_W,dtype='float32' )
rnn_htilda_weight_U = np.array(rnn_htilda_weight_U,dtype='float32' )
rnn_htilda_weight_bias = np.array(rnn_htilda_weight_bias,dtype='float32' )


np.save("gnn_weight.npy",gnn_weight)
gnn_weight = gnn_weight.reshape(-1)

np.save("rnn_update_weight_W.npy",rnn_update_weight_W)
rnn_update_weight_W = rnn_update_weight_W.reshape(-1)

np.save("rnn_update_weight_U.npy",rnn_update_weight_U)
rnn_update_weight_U = rnn_update_weight_U.reshape(-1)

np.save("rnn_update_weight_bias.npy",rnn_update_weight_bias)
rnn_update_weight_bias = rnn_update_weight_bias.reshape(-1)

np.save("rnn_reset_weight_W.npy",rnn_reset_weight_W)
rnn_reset_weight_W = rnn_reset_weight_W.reshape(-1)

np.save("rnn_reset_weight_U.npy",rnn_reset_weight_U)
rnn_reset_weight_U = rnn_reset_weight_U.reshape(-1)

np.save("rnn_reset_weight_bias.npy",rnn_reset_weight_bias)
rnn_reset_weight_bias = rnn_reset_weight_bias.reshape(-1)

np.save("rnn_htilda_weight_W.npy",rnn_htilda_weight_W)
rnn_htilda_weight_W = rnn_htilda_weight_W.reshape(-1)

np.save("rnn_htilda_weight_U.npy",rnn_htilda_weight_U)
rnn_htilda_weight_U = rnn_htilda_weight_U.reshape(-1)

np.save("rnn_htilda_weight_bias.npy",rnn_htilda_weight_bias)
rnn_htilda_weight_bias = rnn_htilda_weight_bias.reshape(-1)


weights_data = []
offset  = 0

gnn_weight = list(gnn_weight)
data_length = len(gnn_weight)
offset += data_length
weights_data += gnn_weight



rnn_update_weight_W = list(rnn_update_weight_W)
data_length = len(rnn_update_weight_W)
offset += data_length
weights_data += rnn_update_weight_W

rnn_update_weight_U = list(rnn_update_weight_U)
data_length = len(rnn_update_weight_U)
offset += data_length
weights_data += rnn_update_weight_U

rnn_update_weight_bias = list(rnn_update_weight_bias)
data_length = len(rnn_update_weight_bias)
offset += data_length
weights_data += rnn_update_weight_bias



rnn_reset_weight_W = list(rnn_reset_weight_W)
data_length = len(rnn_reset_weight_W)
offset += data_length
weights_data += rnn_reset_weight_W

rnn_reset_weight_U = list(rnn_reset_weight_U)
data_length = len(rnn_reset_weight_U)
offset += data_length
weights_data += rnn_reset_weight_U

rnn_reset_weight_bias = list(rnn_reset_weight_bias)
data_length = len(rnn_reset_weight_bias)
offset += data_length
weights_data += rnn_reset_weight_bias



rnn_htilda_weight_W = list(rnn_htilda_weight_W)
data_length = len(rnn_htilda_weight_W)
offset += data_length
weights_data += rnn_htilda_weight_W

rnn_htilda_weight_U = list(rnn_htilda_weight_U)
data_length = len(rnn_htilda_weight_U)
offset += data_length
weights_data += rnn_htilda_weight_U

rnn_htilda_weight_bias = list(rnn_htilda_weight_bias)
data_length = len(rnn_htilda_weight_bias)
offset += data_length
weights_data += rnn_htilda_weight_bias
'''
f = open('EvolveGCN.bin', 'wb')
packed = struct.pack('f'*len(weights_data), *weights_data)
f.write(packed)
f.close()
'''
