# cuLDPC
#### CUDA implementation of LDPC decoding algorithm

## Description
This is a CUDA-based software implementation of LDPC decoding algorithm. The code was developed by the authors for research purpose. 

This detailed explanation of the algorithm can be found from the following papers (you can find them in /doc directory):

> 1. G. Wang, M. Wu, Y. Sun and J. R. Cavallaro, "A massively parallel implementation of QC-LDPC decoder on GPU," 2011 IEEE 9th Symposium on Application Specific Processors (SASP), San Diego, CA, 2011, pp. 82-85.

> 2. G. Wang, M. Wu, B. Yin and J. R. Cavallaro, "High throughput low latency LDPC decoding on GPU for SDR systems," 2013 IEEE Global Conference on Signal and Information Processing (GlobalSIP), Austin, TX, 2013, pp. 1258-1261.

> 3. M. Wu and G. Wang, Massively Parallel Signal Processing for Wireless Communication Systems, GPU Technology Conference (GTC) 2013. March 18-21, 2013, San Jose, California.

## Algorithms
The code implemented Quasi-cyclic LDPC code decoder. The set up is:
* 802.16m WiMax and 802.11n
* Min-sum algorithm and SPA algorithm

## Disclaimer
When the code was developed in 2013, the authors used the following development environment: 
* a PC installing Ubuntu Linux OS
* Intel i7 CPU
* 16GB RAM
* four NVIDIA Titan GPUs and one GTX-690 GPU
* CUDA v4/v5.

The code was debugged based on the above machine. There might be issues directly run the code on older or newer GPUs. Due to the device limitation, we cannot test the code on any other devices. 

The code may not reflect all the optimizations in the paper, but it implemented most of the ideas in the above papers. You can easily tweak the parameters to make the code work for your systems.

The code is provided as is, and it can be used for any purpose, e.g., studying LDPC decoding, perform LDPC-related research, and so on. If you use this code for your research, please cite the papers listed above. 

## License
    Copyright 2016 Guohui Wang

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
