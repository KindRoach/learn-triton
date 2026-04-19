FROM nvcr.io/nvidia/pytorch:26.02-py3

RUN pip3 install triton

# config user and bash completion
RUN apt-get update && apt-get install -y sudo bash-completion \
    && echo "ubuntu ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/ubuntu \
    && chmod 0440 /etc/sudoers.d/ubuntu
