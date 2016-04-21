# cuLDPC
CUDA implementation of LDPC decoding algorithm

## Description
This is a CUDA-based software implementation of LDPC decoding algorithm. The code was developed by the authors for research purpose. 

This detailed explanation of the algorithm can be found from the following papers:
	1. Wang, G., Wu, M., Sun, Y., & Cavallaro, J. R. (2011, June). A massively parallel implementation of QC-LDPC decoder on GPU. In Application Specific Processors (SASP), 2011 IEEE 9th Symposium on (pp. 82-85). IEEE.
	
	2. Wang, G., Wu, M., Yin, B., & Cavallaro, J. R. (2013, December). High throughput low latency LDPC decoding on GPU for SDR systems. In Global Conference on Signal and Information Processing (GlobalSIP), 2013 IEEE (pp. 1258-1261). IEEE.

When the code was developed in 2013, the authors used the following development environment: 
* a PC installing Ubuntu Linux OS
* Intel i7 CPU
* 16GB RAM
* four NVIDIA Titan GPUs

Therefore, the code was debugged based on the above machine. There might be issues directly run the code on older or newer GPUs. Due to the device limitation, we cannot test the code on any other devices. 

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