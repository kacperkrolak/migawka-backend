
# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM ubuntu:latest

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Install production dependencies.
RUN apt-get update
RUN apt-get install -y python3-pip ffmpeg libsm6 libxext6
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip3 install --no-cache-dir -e ./detectron2

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# Timeout is set to 0 to disable the timeouts of the workers to allow Cloud Run to handle instance scaling.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app
#CMD exec gunicorn --bind :8080 --workers 1 --threads 8 --timeout 0 main:app