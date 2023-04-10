

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


gnn_update_weight_W = np.load("../DGNN_parameters/weights/gnn_update_weight_W.npy")
gnn_update_weight_U = np.load("../DGNN_parameters/weights/gnn_update_weight_U.npy")
gnn_update_weight_bias = np.load("../DGNN_parameters/weights/gnn_update_weight_bias.npy")

gnn_reset_weight_W =  np.load("../DGNN_parameters/weights/gnn_reset_weight_W.npy")
gnn_reset_weight_U = np.load("../DGNN_parameters/weights/gnn_reset_weight_U.npy")
gnn_reset_weight_bias = np.load("../DGNN_parameters/weights/gnn_reset_weight_bias.npy")

gnn_htilda_weight_W = np.load("../DGNN_parameters/weights/gnn_htilda_weight_W.npy")
gnn_htilda_weight_U = np.load("../DGNN_parameters/weights/gnn_htilda_weight_U.npy")
gnn_htilda_weight_bias = np.load("../DGNN_parameters/weights/gnn_htilda_weight_bias.npy")

gnn_output_weight_W = np.load("../DGNN_parameters/weights/gnn_output_weight_W.npy")
gnn_output_weight_U = np.load("../DGNN_parameters/weights/gnn_output_weight_U.npy")
gnn_output_weight_bias = np.load("../DGNN_parameters/weights/gnn_output_weight_bias.npy")

gnn_update_weight_W = torch.from_numpy(gnn_update_weight_W )
gnn_update_weight_U = torch.from_numpy(gnn_update_weight_U )
gnn_update_weight_bias = torch.from_numpy(gnn_update_weight_bias )

gnn_reset_weight_W = torch.from_numpy(gnn_reset_weight_W )
gnn_reset_weight_U = torch.from_numpy(gnn_reset_weight_U )
gnn_reset_weight_bias = torch.from_numpy(gnn_reset_weight_bias )

gnn_htilda_weight_W = torch.from_numpy(gnn_htilda_weight_W )
gnn_htilda_weight_U = torch.from_numpy(gnn_htilda_weight_U )
gnn_htilda_weight_bias = torch.from_numpy(gnn_htilda_weight_bias )

gnn_output_weight_W = torch.from_numpy(gnn_output_weight_W )
gnn_output_weight_U = torch.from_numpy(gnn_output_weight_U )
gnn_output_weight_bias = torch.from_numpy(gnn_output_weight_bias )

gnn_update_weight_W = gnn_update_weight_W.to(device)
gnn_update_weight_U = gnn_update_weight_U.to(device)
gnn_update_weight_bias = gnn_update_weight_bias.to(device)

gnn_reset_weight_W = gnn_reset_weight_W.to(device)
gnn_reset_weight_U = gnn_reset_weight_U.to(device)
gnn_reset_weight_bias = gnn_reset_weight_bias.to(device)

gnn_htilda_weight_W = gnn_htilda_weight_W.to(device)
gnn_htilda_weight_U = gnn_htilda_weight_U.to(device)
gnn_htilda_weight_bias = gnn_htilda_weight_bias.to(device)

gnn_output_weight_W = gnn_output_weight_W.to(device)
gnn_output_weight_U = gnn_output_weight_U.to(device)
gnn_output_weight_bias = gnn_output_weight_bias.to(device)



node_embedding = np.load("../DGNN_parameters/node_embedding_wiki/node_embedding.npy")
#dd = node_embedding
edge_embedding = np.load("../DGNN_parameters/edge_embedding_wiki/edge_embedding.npy")
cell_state = np.zeros([3783,67], dtype = 'float32')

node_num = np.load("../DGNN_parameters/node_num/node_num_wiki.npy")
edge_num = np.load("../DGNN_parameters/edge_num/edge_num_wiki.npy")
edge_info = np.load("../DGNN_parameters/edge_info/edge_info.npy")

neighborhood_ref_table = np.load("../DGNN_parameters/renumbering_table/neighborhood_ref_table.npy")
neighborhood_ref_table = np.array(neighborhood_ref_table, dtype = 'int')
neighborhood_reverse_ref_table = np.load("../DGNN_parameters/renumbering_table/neighborhood_reverse_ref_table.npy")



degree_table = np.zeros([137, max(node_num) * 3])
degree_table = np.array(degree_table, dtype =  'int')


neighbor_table = np.zeros([137, max(node_num) * 67 * 2]) 
neighbor_table = np.array(neighbor_table, dtype =  'int')




cell_state_on_chip = np.zeros( [max(node_num),67], dtype = 'float32')
cc_temp = np.zeros( [max(node_num),67], dtype = 'float32')
node_embedding_after_MP = np.zeros([max(node_num),67], dtype = 'float32')
node_embedding_after_MP_temp = np.zeros([max(node_num),67], dtype = 'float32')
edge_embedding_on_chip = np.zeros([1686,67], dtype = 'float32')


node_embedding_after_update = torch.zeros([max(node_num),67])
node_embedding_after_reset = torch.zeros([max(node_num),67])
node_embedding_after_htilda = torch.zeros([max(node_num),67])

node_embedding_after_htilda_temp = torch.zeros([max(node_num),67])

node_embedding_after_output = torch.zeros([max(node_num),67])
#node_embedding_after_multi_1 = torch.zeros([max(node_num),67])
node_embedding_after_multi_2 = torch.zeros([max(node_num),67])

node_embedding_after_update = node_embedding_after_update.to(device)
node_embedding_after_reset = node_embedding_after_reset.to(device)
node_embedding_after_htilda = node_embedding_after_htilda.to(device)

node_embedding_after_htilda_temp = node_embedding_after_htilda_temp.to(device)

node_embedding_after_output = node_embedding_after_output.to(device)
#node_embedding_after_multi_1 = node_embedding_after_multi_1.to(device)
node_embedding_after_multi_2 = node_embedding_after_multi_2.to(device)




graph = np.load("../DGNN_parameters/save_adj_idx_array.npy",allow_pickle=True)
source_node_list  = [[] for i in range(137)]
edge_num_1 = []
node_num_1 = []
for i in range (137):
    edge_num_1.append(graph[i].shape[0])
for i in range(137):
    for j in range(edge_num_1[i]):
        source_node_list[i].append(graph[i][j][0])
source_node_index_list = lists = [[] for i in range(137)]
for i in range(137):
   source_node_index_list[i]  = deleteDuplicatedElementFromList3(source_node_list[i])

node_embedding_on_chip =  np.zeros([max(node_num),67], dtype = 'float32')




total_time_1 = 0 
total_time_2 = 0 
edge_total_num = 0
## prepare the neighborhood information on CPU##
with open('/sys/class/powercap/intel-rapl:1/energy_uj') as f:
    start_energy_uj = int(f.read())
for iter in range(1):
    degree_table = np.zeros([137, max(node_num) * 3])
    degree_table = np.array(degree_table, dtype =  'int')
    total_time_prepare = 0
    for time in range(137):
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
    ## end of preparing the neighborhood information on CPU##        
    
    print("the total time of preparing neighborhood table is:",total_time_prepare)


    time_total = 0
    for time in range(137):
        start_time = mytime.time()
        for nd in range(node_num[time]):
            node_embedding_on_chip[nd] = node_embedding[source_node_index_list[time][nd]]
            #aa = cell_state[source_node_index_list[time][nd]]
            
            cell_state_on_chip[nd] = cell_state[source_node_index_list[time][nd]]
            cc_temp[nd] =  cell_state[source_node_index_list[time][nd]]  
        for nd in range(node_num[time]):
          
          u =  nd   #define the position of the valid node embedding on board
          total_neigh = (degree_table[time][u * 3])
          start_idx =  (degree_table[time][u * 3 + 1])
          
          msg = np.zeros([67], dtype = 'float32')
          
          norm  = np.zeros([67], dtype = 'float32')
          
          for i in range(total_neigh):
              v = (neighbor_table[time][start_idx + i * 2])
              e = (neighbor_table[time][start_idx + i * 2 + 1])
              norm[i] = (1 / (np.sqrt(degree_table[time][u * 3] + 1))) * (1 / (np.sqrt(degree_table[time][v * 3] + 1)))
              msg = msg + norm[i] * (edge_embedding[edge_total_num + e] + node_embedding_on_chip[v])
          node_embedding_after_MP[u] = msg
          #node_embedding_after_MP_temp[u] = msg
          node_embedding_after_MP[u] = node_embedding_after_MP[u] + (1 / (degree_table[time][u * 3] + 1)) * node_embedding_on_chip[u]
          
        node_embedding_after_MP_tensor = torch.from_numpy(node_embedding_after_MP)
        cell_state_on_chip_tensor = torch.from_numpy(cell_state_on_chip)
        node_embedding_after_MP_tensor = node_embedding_after_MP_tensor.to(device)
        cell_state_on_chip_tensor = cell_state_on_chip_tensor.to(device)
        
        start_1 = mytime.time()
        
        node_embedding_after_update[0:node_num[time]] = torch.mm(node_embedding_after_MP_tensor[0:node_num[time]], gnn_update_weight_W) + torch.mm(node_embedding_after_MP_tensor[0:node_num[time]], gnn_update_weight_U) + gnn_update_weight_bias
        node_embedding_after_update[0:node_num[time]] = torch.sigmoid(node_embedding_after_update[0:node_num[time]])
        
        node_embedding_after_reset[0:node_num[time]] = torch.mm(node_embedding_after_MP_tensor[0:node_num[time]], gnn_reset_weight_W) + torch.mm(node_embedding_after_MP_tensor[0:node_num[time]], gnn_reset_weight_U) + gnn_reset_weight_bias
        node_embedding_after_reset[0:node_num[time]] = torch.sigmoid(node_embedding_after_reset[0:node_num[time]])
        
        node_embedding_after_output[0:node_num[time]] = torch.mm(node_embedding_after_MP_tensor[0:node_num[time]], gnn_output_weight_W) + torch.mm(node_embedding_after_MP_tensor[0:node_num[time]], gnn_output_weight_U) + gnn_output_weight_bias
        node_embedding_after_output[0:node_num[time]] = torch.sigmoid(node_embedding_after_output[0:node_num[time]])
        
        node_embedding_after_htilda[0:node_num[time]] = torch.mm(node_embedding_after_MP_tensor[0:node_num[time]], gnn_htilda_weight_W) + torch.mm(node_embedding_after_MP_tensor[0:node_num[time]], gnn_htilda_weight_U) + gnn_htilda_weight_bias
        node_embedding_after_htilda_temp[0:node_num[time]] = node_embedding_after_htilda[0:node_num[time]]
        node_embedding_after_htilda[0:node_num[time]] = torch.tanh(node_embedding_after_htilda[0:node_num[time]])
        
        cell_state_on_chip_tensor[0:node_num[time]] = node_embedding_after_update[0:node_num[time]] * cell_state_on_chip_tensor[0:node_num[time]] + node_embedding_after_reset[0:node_num[time]] * node_embedding_after_htilda[0:node_num[time]]
        
        node_embedding_after_multi_2[0:node_num[time]] = torch.tanh(cell_state_on_chip_tensor[0:node_num[time]]) * node_embedding_after_output[0:node_num[time]]
        
        edge_total_num = edge_total_num + edge_num[time]
        
        
        
        
        
        
        for nd in range(node_num[time]):
            original_position = neighborhood_reverse_ref_table[time][nd]
            node_embedding[original_position] = node_embedding_after_multi_2[nd].cpu().numpy()
            cell_state[original_position] = cell_state_on_chip_tensor[nd].cpu().numpy()
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