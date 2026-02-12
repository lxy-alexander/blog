---
title: "Nvidia System Management Interface"
published: 2026-02-11
description: "Nvidia System Management Interface"
image: ""
tags: ["tools","Nvidia System Management Interface"]
category: tools
draft: false
lang: ""
---



>Nvidia-smi is NVIDIA System Management Interface. It is used to monitor and manage the GPU device status, such as GPU memory usage, GPU utilization, Temperature and power consumption.



## 1) Show overall GPU status

```
nvidia-smi
```

Most commonly used to **quickly check whether GPUs are idle or busy**.

------



## 2) Monitor in real time

Refresh every second:

```
nvidia-smi -l 1
```

------



## 3) List all GPUs

```
nvidia-smi -L
```

------



## 4) Show processes using GPUs

`pmon`: process monitor

```
nvidia-smi pmon
```

Refresh every 2 seconds:

```
nvidia-smi pmon -d 2
```



| Column      | Full name                                      | Meaning                                                      |
| ----------- | ---------------------------------------------- | ------------------------------------------------------------ |
| ==**pid**== | Process ID                                     | The Linux process ID using the GPU.                          |
| **type**    | Process type                                   | Type of GPU workload: • **C** = Compute (CUDA) • **G** = Graphics • **C+G** = Both compute and graphics |
| ==**sm**==  | Streaming Multiprocessor utilization           | Percentage of **GPU compute cores** being used by the process. |
| ==**mem**== | Memory controller utilization                  | Percentage of **GPU memory bandwidth** used by the process.  |
| **enc**     | Encoder utilization                            | Usage of the **NVENC video encoder** by the process.         |
| **dec**     | Decoder utilization                            | Usage of the **NVDEC video decoder** by the process.         |
| **jpg**     | JPEG engine utilization                        | Usage of the **hardware JPEG decoder/encoder**.              |
| **ofa**     | Optical Flow Accelerator utilization           | Usage of the **hardware optical-flow engine** (used in video/vision tasks). |
| ==**fb**==  | Frame Buffer memory                            | Amount of **GPU VRAM used** by the process (usually in MB).  |
| **ccpm**    | Compute & Copy Engine / Protected Memory info* | Internal GPU engine / protection state info; often **0** on many systems and not commonly used in basic monitoring. |

:::important 

pid: show which process is using the GPU

sm: Indicates whether the GPU cores are actively computing.

fb: VRAM/DRAM usage. Indicates how much is used

mem: memory bandwidth utilization 

:::

```
[xli49@ghpc008 ~]$ nvidia-smi pmon -i 0 -s um
# gpu         pid   type     sm    mem    enc    dec    jpg    ofa     fb   ccpm    command 
# Idx           #    C/G      %      %      %      %      %      %     MB     MB    name 
    0          -     -      -      -      -      -      -      -      -      -    -              
    0          -     -      -      -      -      -      -      -      -      -    -              
    0          -     -      -      -      -      -      -      -      -      -                 
^C[xli49@ghpc008 ~]$ nvidia-smi pmon -i 0
# gpu         pid   type     sm    mem    enc    dec    jpg    ofa    command 
# Idx           #    C/G      %      %      %      %      %      %    name 
    0          -     -      -      -      -      -      -      -    -              
    0          -     -      -      -      -      -      -      -    -              
    0          -     -      -      -      -      -      -      -    -              
    0          -     -      -      -      -      -      -      -    -                  
```



------



## 5) Custom query of GPU information

```
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv
```

Commonly used for:

-   scripts
-   logging
-   automated monitoring

------

### 6) Log GPU status to a file

```
nvidia-smi -l 5 -f gpu.log
```

Records GPU information every 5 seconds.
