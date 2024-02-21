# Use the NVIDIA TensorFlow base image
FROM nvcr.io/nvidia/tensorflow:23.12-tf2-py3

# Set the working directory
WORKDIR /FinalProjectDOCKER

ADD requirements.txt /FinalProjectDOCKER
# Install dependencies (if needed, continue manually)
RUN pip install --upgrade pip
RUN apt-get update
RUN apt-get install -y libdbus-1-dev libglib2.0-dev \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
RUN apt-get update && apt-get install libgl1 -y
RUN apt-get update && \
    apt-get install -y x11-apps
RUN pip install pydicom
RUN pip install scikit-image
RUN pip install matplotlib
RUN pip install ipywidgets
RUN pip install opencv-python
RUN pip install tables
RUN pip install nibabel
RUN pip install -r /FinalProjectDOCKER/requirements.txt

ENV DISPLAY=:0


# Expose port 8080
EXPOSE 8080

# Specify the command to run on container start - you can change it to execute the script at start - but it is safer to just execute it yourself in case any additional actions need to be performed manually
CMD ["bash"]
