FROM ubuntu:24.04

# install Intel GPU runtime libs
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository -y ppa:kobuk-team/intel-graphics
RUN apt-get install -y libze-intel-gpu1 libze1 intel-metrics-discovery intel-opencl-icd clinfo intel-gsc libze-dev intel-ocloc

# Install Python and pip
RUN apt-get install -y python3 python3-pip python-is-python3

# Install Pytorch with XPU support
RUN rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED
RUN pip install \
    torch==2.9.1 \
    --index-url https://download.pytorch.org/whl/xpu

# install vtune
WORKDIR /vtune
RUN apt-get install -y wget
RUN wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/a04c89ad-d663-4f70-bd3d-bb44f5c16d57/intel-vtune-2025.7.0.248_offline.sh
RUN bash intel-vtune-2025.7.0.248_offline.sh -a -s --eula accept

# setup metrics-discovery for vtune
RUN apt-get update && apt-get install -y libdrm-dev cmake build-essential git

WORKDIR /metrics-discovery

RUN git clone https://github.com/intel/metrics-discovery.git .

RUN cmake -S . -B build && cmake --build build -j && cmake --install build