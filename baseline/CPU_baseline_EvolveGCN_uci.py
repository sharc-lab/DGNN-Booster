

import torch
import numpy as np

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


device = 'cpu'
time_step = 10

gnn_weight = np.load("../DGNN_parameters/EvolveGCN_weights/gnn_weight.npy")


rnn_update_weight_W = np.load("../DGNN_parameters/EvolveGCN_weights/rnn_update_weight_W.npy")
rnn_update_weight_U = np.load("../DGNN_parameters/EvolveGCN_weights/rnn_update_weight_U.npy")
rnn_update_weight_bias = np.load("../DGNN_parameters/EvolveGCN_weights/rnn_update_weight_bias.npy")


rnn_reset_weight_W = np.load("../DGNN_parameters/EvolveGCN_weights/rnn_reset_weight_W.npy")
rnn_reset_weight_U = np.load("../DGNN_parameters/EvolveGCN_weights/rnn_reset_weight_U.npy")
rnn_reset_weight_bias = np.load("../DGNN_parameters/EvolveGCN_weights/rnn_reset_weight_bias.npy")


rnn_htilda_weight_W = np.load("../DGNN_parameters/EvolveGCN_weights/rnn_htilda_weight_W.npy")
rnn_htilda_weight_U = np.load("../DGNN_parameters/EvolveGCN_weights/rnn_htilda_weight_U.npy")
rnn_htilda_weight_bias = np.load("../DGNN_parameters/EvolveGCN_weights/rnn_htilda_weight_bias.npy")


gnn_weights_updated = torch.zeros([11,67,122])
gnn_weights_updated[0] = torch.from_numpy(gnn_weight)

rnn_update_weight_W = torch.from_numpy(rnn_update_weight_W)
rnn_update_weight_U = torch.from_numpy(rnn_update_weight_U)
rnn_update_weight_bias = torch.from_numpy(rnn_update_weight_bias)

rnn_reset_weight_W = torch.from_numpy(rnn_reset_weight_W)
rnn_reset_weight_U = torch.from_numpy(rnn_reset_weight_U)
rnn_reset_weight_bias = torch.from_numpy(rnn_reset_weight_bias)

rnn_htilda_weight_W = torch.from_numpy(rnn_htilda_weight_W)
rnn_htilda_weight_U = torch.from_numpy(rnn_htilda_weight_U)
rnn_htilda_weight_bias = torch.from_numpy(rnn_htilda_weight_bias)


gnn_weights_updated = gnn_weights_updated.to(device)

rnn_update_weight_W = rnn_update_weight_W.to(device)
rnn_update_weight_U = rnn_update_weight_U.to(device)
rnn_update_weight_bias = rnn_update_weight_bias.to(device)

rnn_reset_weight_W = rnn_reset_weight_W.to(device)
rnn_reset_weight_U = rnn_reset_weight_U.to(device)
rnn_reset_weight_bias = rnn_reset_weight_bias.to(device)

rnn_htilda_weight_W = rnn_htilda_weight_W.to(device)
rnn_htilda_weight_U = rnn_htilda_weight_U.to(device)
rnn_htilda_weight_bias = rnn_htilda_weight_bias.to(device)

node_embedding = np.load("../DGNN_parameters/uci/node_embedding.npy")
#dd = node_embedding
edge_embedding = np.load("../DGNN_parameters/uci/edge_embedding.npy")

node_num = np.load("../DGNN_parameters/uci/node_number.npy")
edge_num = np.load("../DGNN_parameters/uci/edge_number.npy")
edge_info = np.load("../DGNN_parameters/uci/graph_info_uci.npy")

neighborhood_ref_table = np.load("../DGNN_parameters/uci/neighborhood_ref_table.npy")
neighborhood_ref_table = np.array(neighborhood_ref_table, dtype = 'int')
neighborhood_reverse_ref_table = np.load("../DGNN_parameters/uci/neighborhood_reverse_ref_table.npy")
neighborhood_reverse_ref_table = np.array(neighborhood_reverse_ref_table, dtype = 'int')



degree_table = np.zeros([192, max(node_num) * 3])
degree_table = np.array(degree_table, dtype =  'int')


neighbor_table = np.zeros([192, max(node_num) * 160 * 2]) 
neighbor_table = np.array(neighbor_table, dtype =  'int')

node_embedding_on_chip =  np.zeros([time_step, max(node_num),67], dtype = 'float32')




source_node_list  = [[] for i in range(192)]
edge_num_1 = []
node_num_1 = []
for i in range (192):
    edge_num_1.append(edge_info[i].shape[0])
for i in range(192):
    for j in range(edge_num_1[i]):
        source_node_list[i].append(edge_info[i][j][0])
source_node_index_list = lists = [[] for i in range(192)]
for i in range(192):
   source_node_index_list[i]  = deleteDuplicatedElementFromList3(source_node_list[i])



with open('/sys/class/powercap/intel-rapl:1/energy_uj') as f:
    start_energy_uj = int(f.read())

total_time_prepare = 0 
### prepare the neighborhood table###
for time in range(192):
        #degree_table = np.zeros([137, max(node_num) * 3])
        #degree_table = np.array(degree_table, dtype =  'int')
    prepare_start = mytime.time()   
    for e in range(edge_num[time]):
        u = (neighborhood_ref_table[time][edge_info[time][e][0]])  #source node idx in the on-chip buffer
        v = (neighborhood_ref_table[time][edge_info[time][e][1]])  #target node idx in thee on-chip buffer
        degree_table[time][u * 3] = degree_table[time][u * 3] + 1
    for n in torch.arange(1,node_num[time],1):
        #last_position = valid_list[time][n - 1]
        #position = valid_list[time][n]
        degree_table[time][n * 3 + 1] = degree_table[time][(n - 1) * 3] * 2 + degree_table[time][(n - 1) * 3 + 1]
    for e in range(edge_num[time]):
        u = (neighborhood_ref_table[time][edge_info[time][e][0]])  #source node idx in the on-chip buffer
        v = (neighborhood_ref_table[time][edge_info[time][e][1]])  #target node idx in thee on-chip buffer
        
        total_neigh = degree_table[time][u * 3]
        start_idx =  (degree_table[time][u * 3 + 1])
        offset_idx =  (degree_table[time][u * 3 + 2])
        
        neighbor_table[time][start_idx + offset_idx] = v
        neighbor_table[time][start_idx + offset_idx + 1] = e
        degree_table[time][u * 3 + 2] = degree_table[time][u * 3 + 2] + 2
        
    prepare_end = mytime.time()
    total_time_prepare = total_time_prepare + (prepare_end - prepare_start)
    
print("the total time of preparing neighborhood table is:",total_time_prepare)
        
node_embedding_after_MP = np.zeros([time_step, max(node_num),67], dtype = 'float32')

node_embedding_updated = np.zeros([time_step, max(node_num), 122], dtype = 'float32')
node_embedding_updated = torch.from_numpy(node_embedding_updated)

gnn_weights_after_update = torch.zeros([time_step, 67,122])
gnn_weights_after_reset = torch.zeros([time_step, 67, 122])
gnn_weights_after_htilda = torch.zeros([time_step, 67, 122])
gnn_weights_after_multi_1 = torch.zeros([time_step, 67, 122])
gnn_weights_after_multi_2 = torch.zeros([time_step, 67, 122])

gnn_weights_after_update = gnn_weights_after_update.to(device)
gnn_weights_after_reset = gnn_weights_after_reset.to(device)
gnn_weights_after_htilda = gnn_weights_after_htilda.to(device)
gnn_weights_after_multi_1 = gnn_weights_after_multi_1.to(device)
gnn_weights_after_multi_2 = gnn_weights_after_multi_2.to(device)
node_embedding_updated = node_embedding_updated.to(device)

node_embedding_final = np.zeros([max(node_num), 122], dtype = 'float32')

time_total_1 = 0
edge_total_num = 0

time_total_2 = 0
total_neigh_list = []

time_total = 0
for step in range (183):
    edge_total_num_inner = 0
    start_time = mytime.time()
    for time in range (time_step):
        for nd in range (node_num[step + time]):
            node_embedding_on_chip[time][nd] = node_embedding[source_node_index_list[step + time][nd]]
        for nd in range(node_num[step + time]):
          
              u =  nd   #define the position of the valid node embedding on board
              total_neigh = (degree_table[step + time][u * 3])
              start_idx =  (degree_table[step + time][u * 3 + 1])
              
              msg = np.zeros([67], dtype = 'float32')
              
              norm  = np.zeros([160], dtype = 'float32')
              total_neigh_list.append(total_neigh)
              for i in range(total_neigh):
                  v = (neighbor_table[step + time][start_idx + i * 2])
                  e = (neighbor_table[step + time][start_idx + i * 2 + 1])
                  norm[i] = (1 / (np.sqrt(degree_table[step + time][u * 3] + 1))) * (1 / (np.sqrt(degree_table[step + time][v * 3] + 1)))
                  msg = msg + norm[i] * (edge_embedding[edge_total_num + edge_total_num_inner + e] + node_embedding_on_chip[time][v])
              node_embedding_after_MP[time][u] = msg
              node_embedding_after_MP[time][u] = node_embedding_after_MP[time][u] + (1 / (degree_table[step + time][u * 3] + 1)) * node_embedding_on_chip[time][u]
        edge_total_num_inner = edge_total_num_inner + edge_num[step + time]
          
    node_embedding_after_MP_tensor = torch.from_numpy(node_embedding_after_MP)
    #cell_state_on_chip_tensor = torch.from_numpy(cell_state_on_chip)
    node_embedding_after_MP_tensor = node_embedding_after_MP_tensor.to(device)
    #cell_state_on_chip_tensor = cell_state_on_chip_tensor.to(device)
    edge_total_num = edge_total_num + edge_num[step]
    
    for time in range (time_step):
        gnn_weights_after_update[time] = torch.mm(rnn_update_weight_W, gnn_weights_updated[time]) + torch.mm(rnn_update_weight_U, gnn_weights_updated[time]) + rnn_update_weight_bias
        #gnn_weights_after_update[time] = 1.0/(1.0 + np.exp(-gnn_weights_after_update[time]))
        gnn_weights_after_update[time]  = torch.sigmoid(gnn_weights_after_update[time])
        
        gnn_weights_after_reset[time] = torch.mm(rnn_reset_weight_W, gnn_weights_updated[time]) + torch.mm(rnn_reset_weight_U, gnn_weights_updated[time]) + rnn_reset_weight_bias
        #gnn_weights_after_reset[time] = 1.0/(1.0 + np.exp(-gnn_weights_after_reset[time]))
        gnn_weights_after_reset[time] = torch.sigmoid(gnn_weights_after_reset[time])
        
        gnn_weights_after_multi_1[time] = gnn_weights_updated[time] * gnn_weights_after_reset[time]
        
        
        gnn_weights_after_htilda[time] = torch.mm(rnn_htilda_weight_W, gnn_weights_updated[time]) + torch.mm(rnn_htilda_weight_U, gnn_weights_after_multi_1[time]) + rnn_htilda_weight_bias
        #gnn_weights_after_htilda[time] = (np.exp(gnn_weights_after_htilda[time]) - np.exp(-gnn_weights_after_htilda[time]))/(np.exp(gnn_weights_after_htilda[time]) + np.exp(-gnn_weights_after_htilda[time]))
        gnn_weights_after_htilda[time] = torch.tanh(gnn_weights_after_htilda[time])
        
        gnn_weights_updated[time + 1] = (1.0 - gnn_weights_after_update[time]) * gnn_weights_updated[time] + gnn_weights_after_update[time] * gnn_weights_after_htilda[time]
        
        node_embedding_updated[time][0: node_num[time + step]] = torch.mm(node_embedding_after_MP_tensor[time][0: node_num[time + step]],gnn_weights_updated[time + 1] )
        
    node_embeeding_final = node_embedding_updated[-1][0: node_num[time + step]].cpu().numpy()
    
    
    end_time = mytime.time()
    time_total = time_total +  end_time - start_time
total =  time_total +  total_time_prepare 
with open('/sys/class/powercap/intel-rapl:1/energy_uj') as f:
    end_energy_uj = int(f.read())
total_energy = (end_energy_uj - start_energy_uj) / 1e6
power = total_energy / total
print("the value of inference is:",time_total)         
print("the value of time total is:",total)
print("the power is:",power)