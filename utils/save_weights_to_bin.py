

import torch
import numpy as np
import struct

def deleteDuplicatedElementFromList3(listA):
        #return list(set(listA))
        return sorted(set(listA), key = listA.index)



## this is the weights for convLATM  ###

np.random.seed(100)
gnn_update_weight_W = 2 *np.random.rand(67,67) - 1
gnn_update_weight_U = 2 *np.random.rand(67,67) - 1
gnn_update_weight_bias = 2 *np.random.rand(67) - 1

gnn_reset_weight_W = 2 *np.random.rand(67,67) - 1
gnn_reset_weight_U = 2 *np.random.rand(67,67) - 1
gnn_reset_weight_bias = 2 *np.random.rand(67) - 1

gnn_htilda_weight_W = 2 *np.random.rand(67,67) - 1
gnn_htilda_weight_U = 2 *np.random.rand(67,67) - 1
gnn_htilda_weight_bias = 2 *np.random.rand(67) - 1

gnn_output_weight_W = 2 *np.random.rand(67,67) - 1
gnn_output_weight_U = 2 *np.random.rand(67,67) - 1
gnn_output_weight_bias = 2 *np.random.rand(67) - 1

gnn_update_weight_W = np.array(gnn_update_weight_W, dtype = 'float32')
np.save("gnn_update_weight_W.npy",gnn_update_weight_W)
gnn_update_weight_W = gnn_update_weight_W.reshape(-1)

gnn_update_weight_U = np.array(gnn_update_weight_U, dtype = 'float32')
np.save("gnn_update_weight_U.npy",gnn_update_weight_U)
gnn_update_weight_U = gnn_update_weight_U.reshape(-1)

gnn_update_weight_bias = np.array(gnn_update_weight_bias, dtype = 'float32')
np.save("gnn_update_weight_bias.npy",gnn_update_weight_bias)
gnn_update_weight_bias = gnn_update_weight_bias.reshape(-1)


gnn_reset_weight_W = np.array(gnn_reset_weight_W, dtype = 'float32')
np.save("gnn_reset_weight_W.npy",gnn_reset_weight_W)
gnn_reset_weight_W = gnn_reset_weight_W.reshape(-1)

gnn_reset_weight_U = np.array(gnn_reset_weight_U, dtype = 'float32')
np.save("gnn_reset_weight_U.npy",gnn_reset_weight_U)
gnn_reset_weight_U = gnn_reset_weight_U.reshape(-1)

gnn_reset_weight_bias = np.array(gnn_reset_weight_bias, dtype = 'float32')
np.save("gnn_reset_weight_bias.npy",gnn_reset_weight_bias)
gnn_reset_weight_bias = gnn_reset_weight_bias.reshape(-1)


gnn_htilda_weight_W = np.array(gnn_htilda_weight_W, dtype = 'float32')
np.save("gnn_htilda_weight_W.npy",gnn_htilda_weight_W)
gnn_htilda_weight_W = gnn_htilda_weight_W.reshape(-1)

gnn_htilda_weight_U = np.array(gnn_htilda_weight_U, dtype = 'float32')
np.save("gnn_htilda_weight_U.npy",gnn_htilda_weight_U)
gnn_htilda_weight_U = gnn_htilda_weight_U.reshape(-1)

gnn_htilda_weight_bias = np.array(gnn_htilda_weight_bias, dtype = 'float32')
np.save("gnn_htilda_weight_bias.npy",gnn_htilda_weight_bias)
gnn_htilda_weight_bias = gnn_htilda_weight_bias.reshape(-1)


gnn_output_weight_W = np.array(gnn_output_weight_W, dtype = 'float32')
np.save("gnn_output_weight_W.npy",gnn_output_weight_W)
gnn_output_weight_W = gnn_output_weight_W.reshape(-1)

gnn_output_weight_U = np.array(gnn_output_weight_U, dtype = 'float32')
np.save("gnn_output_weight_U.npy",gnn_output_weight_U )
gnn_output_weight_U = gnn_output_weight_U.reshape(-1)

gnn_output_weight_bias = np.array(gnn_output_weight_bias, dtype = 'float32')
np.save("gnn_output_weight_bias.npy",gnn_output_weight_bias )
gnn_output_weight_bias = gnn_output_weight_bias.reshape(-1)

'''
weight_tensor = dict2.mlp[0].weight.data
offset  = 0
weights_data = []
for i in range(22):
    data = list(dict1._parameters[i].cpu().detach().view(-1).numpy())
    data_length = len(data)
    offset += data_length
    weights_data += data
'''    

weights_data = []
offset  = 0

gnn_update_weight_W = list(gnn_update_weight_W)
data_length = len(gnn_update_weight_W)
offset += data_length
weights_data += gnn_update_weight_W

gnn_update_weight_U = list(gnn_update_weight_U)
data_length = len(gnn_update_weight_U)
offset += data_length
weights_data += gnn_update_weight_U

gnn_update_weight_bias = list(gnn_update_weight_bias)
data_length = len(gnn_update_weight_bias)
offset += data_length
weights_data += gnn_update_weight_bias






gnn_reset_weight_W = list(gnn_reset_weight_W)
data_length = len(gnn_reset_weight_W)
offset += data_length
weights_data += gnn_reset_weight_W

gnn_reset_weight_U = list(gnn_reset_weight_U)
data_length = len(gnn_reset_weight_U)
offset += data_length
weights_data += gnn_reset_weight_U

gnn_reset_weight_bias = list(gnn_reset_weight_bias)
data_length = len(gnn_reset_weight_bias)
offset += data_length
weights_data += gnn_reset_weight_bias







gnn_htilda_weight_W = list(gnn_htilda_weight_W)
data_length = len(gnn_htilda_weight_W)
offset += data_length
weights_data += gnn_htilda_weight_W

gnn_htilda_weight_U = list(gnn_htilda_weight_U)
data_length = len(gnn_htilda_weight_U)
offset += data_length
weights_data += gnn_htilda_weight_U

gnn_htilda_weight_bias = list(gnn_htilda_weight_bias)
data_length = len(gnn_htilda_weight_bias)
offset += data_length
weights_data += gnn_htilda_weight_bias




gnn_output_weight_W = list(gnn_output_weight_W)
data_length = len(gnn_output_weight_W)
offset += data_length
weights_data += gnn_output_weight_W

gnn_output_weight_U = list(gnn_output_weight_U)
data_length = len(gnn_output_weight_U)
offset += data_length
weights_data += gnn_output_weight_U

gnn_output_weight_bias = list(gnn_output_weight_bias)
data_length = len(gnn_output_weight_bias)
offset += data_length
weights_data += gnn_output_weight_bias

'''
f = open('convLSTM.bin', 'wb')
packed = struct.pack('f'*len(weights_data), *weights_data)
f.write(packed)
f.close()
'''


node_embedding = 2 *np.random.rand(3783,67) - 1
node_embedding = np.array(node_embedding, dtype = 'float32')
cc = node_embedding
np.save("node_embedding.npy",node_embedding)
node_embedding = node_embedding.reshape(-1)
node_embedding = list(node_embedding)
data_length = len(node_embedding)
'''
f = open('node_embedding_wiki.bin', 'wb')
packed = struct.pack('f'*len(node_embedding), *node_embedding)
f.write(packed)
f.close()
'''
edge_embedding = 2 *np.random.rand(31792,67) - 1
edge_embedding_1 = edge_embedding
edge_embedding = np.array(edge_embedding, dtype = 'float32')
np.save("edge_embedding.npy",edge_embedding)
edge_embedding = edge_embedding.reshape(-1)
edge_embedding = list(edge_embedding)
data_length = len(edge_embedding)

'''
f = open('edge_embedding_wiki.bin', 'wb')
packed = struct.pack('f'*len(edge_embedding), *edge_embedding)
f.write(packed)
f.close()
'''

graph = np.load(r"C:\Users\chenhanqiou\Desktop\visual_studio_code_GenGNN\EvolveGCN-master (1)\EvolveGCN-master\graph_information\save_adj_idx_array.npy",allow_pickle=True)
source_node_list  = [[] for i in range(137)]
edge_num = []
node_num = []
for i in range (137):
    edge_num.append(graph[i].shape[0])
edge_num = np.array(edge_num, dtype = 'int')
np.save("edge_num_wiki.npy", edge_num)
edge_number = np.array(edge_num, dtype = 'float32')    
edge_number = list(edge_number)
data_length  = len(edge_number)    

'''
f = open('edge_num_wiki.bin', 'wb')
packed = struct.pack('f'*len(edge_number), *edge_number)
f.write(packed)
f.close()
'''


for i in range(137):
    for j in range(edge_num[i]):
        source_node_list[i].append(graph[i][j][0])
source_node_index_list = lists = [[] for i in range(137)]
for i in range(137):
   source_node_index_list[i]  = deleteDuplicatedElementFromList3(source_node_list[i])
for i in range(137):
    node_num.append(len(source_node_index_list[i]))
    
node_number = np.array(node_num, dtype = 'float32')
node_num = np.array(node_number, dtype = 'int')
np.save("node_num_wiki.npy", node_num)
node_number = list(node_number)
data_length = len(node_number)

'''
f = open('node_num_wiki.bin', 'wb')
packed = struct.pack('f'*len(node_number), *node_number)
f.write(packed)
f.close()
'''


edge_info = []
edge_info_array = np.zeros([137,1686,2], dtype = 'int')

for i in range (137):
    for j in range(edge_num[i]):
        edge_info_array[i][j] = graph[i][j]

np.save("edge_info.npy", edge_info_array)
edge_info_array = np.array(edge_info_array, dtype = 'float32')

        
for i in range (137):
    for j in range(edge_num[i]):  
        for k in range(2):
            edge_info.append(edge_info_array[i][j][k])
'''
for i in range (137):
    edge_info += list( (graph[i].reshape(-1)))
'''
data_length = len(edge_info)

'''
f = open('edge_info.bin', 'wb')
packed = struct.pack('f'*len(edge_info), *edge_info)
f.write(packed)
f.close()
'''


node_embedding_list = [np.zeros([node_num[i],67], dtype = 'float32') for i in range(137)]
 #define the status of the node embedding buffer on chip
edge_embedding_list = [np.zeros([edge_num[i],67], dtype = 'float32') for i in range(137)]

for i in range(137):
    for j in range(node_num[i]):
        node_embedding_list[i][j] = node_embedding[source_node_index_list[i][j]]

total_edge_num = 0
for i in range(137):
    total_edge_num+= len(source_node_list[i])
edge_embedding = 2 *np.random.rand(total_edge_num,67) - 1
edge_embedding = np.array(edge_embedding, dtype = 'float32')

counter = 0
for i in range(137):
    for j in range(edge_num[i]):
        edge_embedding_list[i][j] = edge_embedding[counter + j]
    counter = counter + edge_num[i]
## graph preprocessing ends  ### 

##prepare neighborhood renumbering tabl##
neighborhood_ref_table = np.zeros([137,3783])
neighborhood_reverse_ref_table = [np.zeros(node_num[i]) for i in range(137)]
for i in range(137):
    for j in range(node_num[i]):
        neighborhood_ref_table[i][source_node_index_list[i][j]] = j

for i in range(137):
    for j in range(node_num[i]):
        neighborhood_reverse_ref_table[i][j] = source_node_index_list[i][j]  
        
neighborhood_ref_table = np.array(neighborhood_ref_table, dtype = 'float32')
np.save("neighborhood_ref_table.npy",neighborhood_ref_table)        
neighborhood_ref_table =  neighborhood_ref_table.reshape(-1)  
neighborhood_ref_table = list(neighborhood_ref_table) 
'''
f = open('neighborhood_ref_table.bin', 'wb')
packed = struct.pack('f'*len(neighborhood_ref_table), *neighborhood_ref_table)
f.write(packed)
f.close()
'''

    

        
neighborhood_reverse_ref_table_array = np.zeros([137,578], dtype  = 'int')

neighborhood_reverse_ref_table_list = []
for i in range (137):
    for j in range (node_num[i]):
        neighborhood_reverse_ref_table_array[i][j] = neighborhood_reverse_ref_table[i][j]
np.save("neighborhood_reverse_ref_table.npy",neighborhood_reverse_ref_table_array)
neighborhood_reverse_ref_table_array = np.array(neighborhood_reverse_ref_table_array, dtype = 'float32')



for i in range (137):
    for j in range(578):
        neighborhood_reverse_ref_table_list.append(neighborhood_reverse_ref_table_array[i][j])
'''        
f = open('neighborhood_reverse_ref_table.bin', 'wb')
packed = struct.pack('f'*len(neighborhood_reverse_ref_table_list), *neighborhood_reverse_ref_table_list)
f.write(packed)
f.close()
'''
