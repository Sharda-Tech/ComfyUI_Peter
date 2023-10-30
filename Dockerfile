# Use the official Python image as the base image
FROM nvidia/cuda:11.0.3-base-ubuntu20.04

# Run update
RUN apt update

RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

#Install Python

RUN apt-get install -y python3 
RUN apt-get install -y python3-pip
RUN apt-get install -y git

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone the required GitHub repositories and install dependencies
# Install PyTorch and its dependencies
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Clone the first GitHub repository
RUN git clone https://github.com/facebookresearch/detectron2.git

# Install the detectron2 project from the GitHub repository
RUN pip install git+https://github.com/facebookresearch/detectron2@main#subdirectory=projects/DensePose

# Clone the second GitHub repository
RUN git clone https://github.com/Whizz-Technologies/detectron2-barinov

# Change the working directory to the DensePose project
WORKDIR /app/detectron2-barinov/projects/DensePose

# Copy the model_final_0ed407.pkl file
COPY './model_final_0ed407.pkl' ./

# Install iglovikov_helper_functions
RUN pip install iglovikov_helper_functions

# Install ultralytics
RUN pip install ultralytics

# Change the working directory back to the root
WORKDIR /app

# Clone the third GitHub repository
RUN git clone https://github.com/Sharda-Tech/huggingface-cloth-segmentation

# Change the working directory to the huggingface-cloth-segmentation project
WORKDIR /app/huggingface-cloth-segmentation

# Install the requirements for the project
RUN pip install -r requirements.txt

# Create a model directory
RUN mkdir -p '/app/huggingface-cloth-segmentation/model'

# Copy the cloth_segm.pth file
COPY './cloth_segm.pth' '/app/huggingface-cloth-segmentation/model'

# Set the default command to be executed when the container starts
CMD ["/bin/bash"]
