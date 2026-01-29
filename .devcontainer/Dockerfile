# foundation : python image 
FROM mcr.microsoft.com/devcontainers/python:3.12

# Install system dependencies (as root)
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*
