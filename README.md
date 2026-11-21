# Verilator-X
Verilator-X is a Hybrid-Abstraction RTL Simulation Framework for Neural Network Accelerators. This architecture integrates two synergistic techniques: dynamic mapping-aware modeling (DMM), which identifies layer structures and establishes optimized mappings to processing element (PE) arrays, and precision-driven task scoring (PTS), which evaluates layer criticality and selectively replaces non-critical operations with transaction-level simulation (TLM), thereby reducing host simulation time per cycle with negligible accuracy loss. Verilator-X provides both a precise mode for functional verification and an acceleration mode for fast performance estimation.

<img width="702" height="304" alt="image" src="https://github.com/user-attachments/assets/9d0ab706-c0e2-40c0-9ac5-5d8df8dc44bb" />

Verilator-X achieves efficient simulation at the software level. It bridges the input module and the output module. The basic idea of Verilator-X is to conduct intelligent simulation of the perception neural network layer and hardware structure, and provide two simulation models for users to choose from in different scenarios.

1.Requirements
```
(1) ubuntu >= 20.04 (Recommended)
(2) g++ >= 9.4.0 (Recommended)
(3) Verilator >=5.033 (Recommended)
(4) Python3
```

2.Environment Configuration
```
touch setup.sh    #Create setup.sh
pwd
export VERILATOR_ROOT = /home/***/verilator_tools/verilator #Set the root directory
```

3.Getting Started
It explains how to build and simulate neural network accelerators by the Verilator-X method.

(1) Create the project directory structure.<br>
```
verilator-X/
├── rtl/                             
│   ├── conv_pe_array.v               
│   ├── pool_layer.v                 
│   ├── residual_add.v                
│   ├── depthwise_conv.v             
│   ├── vgg16_accelerator.v
│   ├── resnet50_accelerator.v
│   ├── alexnet_accelerator.v
│   ├── mobilenetv2_accelerator.v
│   └── googlenet_accelerator.v
├── tb/                               
│   ├── tb_generic.cpp               
│   ├── network_desc.h              
│   ├── dmm_scheduler.h/.cpp         
│   ├── dmm_tree_generated.h/.cpp     
│   ├── pts_scorer.h/.cpp          
│   └── tl_layers.h/.cpp             
├── scripts/
│   └── dmm_train.py                
├── data/
│   ├── train.csv                    
│   └── test.csv                     
├── run_all.sh                      
└── README.md
```


(2)RTL Preparation.<br>

The RTL contains all Verilog source files, and the key information includes layer types, PE array size, weight storage structure, etc.<br>


(3)TB Preparation.<br>

TB stores C++ testbench files, DMM, and PTS files, and undertakes all the acceleration logic of Verilator-X. In TB, two major mechanisms of Verilator-X are implemented. The first one is DMM (Dynamic Mapping-Aware Modeling): it identifies the structure of the neural network layer, selects the dataflow template, and generates "task-to-PE mapping". This "mapping" in the Verilator process is not the physical connection of RTL, but how the C++ testbench organizes tiles, inputs and outputs, loop boundaries, and weight loading sequence. The second one is PTS (Precision-driven Task Scoring): it identifies non-critical tasks and replaces cycle-accurate simulation with transaction-level simulation.<br>

tb_generic.cpp is the unified testbench core, the main program of Verilator-X, which accomplishes initialization, DMM decision-making, PTS determination, etc.<br>
```
//PTS Main Loop (Part)
while (!dut->done && compute_cycles < MAX_CYCLES) {
    if (dut->pts_mode && dut->wait_pts) {
        int li = dut->pts_layer_idx;
        if (skip[li]) {
            PtsCtx c = dbg->read_pts(li);
            std::vector<uint8_t> y;
            if (c.is_residual)
                tl_residual_add(c.a.data(), c.b.data(), y.data(), c.h, c.ch);
            else
                tl_max_pool_2x2(c.a.data(), y.data(), c.h, c.ch);
            dbg->write_pts(li, y);
        }
        dut->pts_ready = 1; clock_cycle(dut); dut->pts_ready = 0;
    } else {
        clock_cycle(dut); ++compute_cycles;
    }
}
```
The network_desc.h file defines the LayerDesc structure, which serves as the input data for DMM decisions and PTS judgments.<br>

dmm_scheduler.h/.cpp, DMM Scheduler, includes a rule engine and a decision tree.<br>

The pts_scorer.h/.cpp file defines the PTS scoring, calculating the criticality score for each layer. Layers with a score ≥ θ are marked as "non-critical" and can be replaced at the transaction level.<br>

(4)Train the DMM decision tree.<br>
For the first run, it is necessary to generate the DMM decision tree and training/testing data.
```
cd verilator-X
python3 -m py_compile scripts/dmm_train.py
python3 scripts/dmm_train.py
```
(5)Construction and operation.<br>
```
chmod +x run_all.sh
./run_all.sh
```

Verilator-X also supports running a single neural network accelerator.
```
rm -rf obj_vgg sim_VGG16_*.log  #remove the history files
verilator -Wall --cc -O2 -Wno-fatal --top-module vgg16_accelerator \
    -Mdir obj_vgg \
    rtl/vgg16_accelerator.v rtl/conv_pe_array.v rtl/pool_layer.v \
    --exe tb/tb_generic.cpp tb/dmm_scheduler.cpp tb/dmm_tree_generated.cpp \
           tb/pts_scorer.cpp tb/tl_layers.cpp \
    -CFLAGS "-I$(pwd)/tb -DNETWORK_VGG16" \
    --build -j 4
```

(6)Running results.<br>
We measure total host execution time and total simulation cycles per workload, defining their ratio as kilocycles per second (KCPS). Higher KCPS indicates faster simulation and shorter per-evaluation latency. We also report normalized speedup (SU) over baselines. The hardware is equipped with a PE array of size 64x64 and necessary
hardware structures. <br>

<img width="710" height="323" alt="image" src="https://github.com/user-attachments/assets/7be61963-2e21-47d5-8fe4-5d4d79948afb" />


Meanwhile, Verilator-X generates detailed logs.<br>

```
sim_baseline.log
sim_pts.log
sim_dmm.log
sim_pts_dmm.log
```


