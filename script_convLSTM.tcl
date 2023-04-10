open_project convLSTM_final_100
set_top GCN_RNN_compute_one_graph

add_files dcl.h
add_files DGNN-booster2-GCRN-M2.cpp

open_solution "solution3" -flow_target vivado
set_part {xczu9eg-ffvb1156-2-e}
create_clock -period 10 -name default

## C simulation
# Use Makefile instead. This is even slower.
#csim_design -O -clean

## C code synthesis to generate Verilog code
csynth_design

## C and Verilog co-simulation
## This usually takes a long time so it is commented
## You may uncomment it if necessary
#cosim_design

## export synthesized Verilog code
export_design -format ip_catalog

exit