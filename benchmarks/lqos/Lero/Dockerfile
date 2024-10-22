# Use an official Ubuntu 22.04 LTS as a parent image
FROM ubuntu:22.04

# Set the maintainer label
LABEL maintainer="yourname@example.com"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y ca-certificates

# Install certificates and update package lists
RUN apt-get update && \
    apt-get install -y --no-install-recommends --allow-unauthenticated \
    ca-certificates \
    curl \
    gnupg \
    build-essential \
    libreadline-dev \
    zlib1g-dev \
    vim \
    wget \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /usr/src/app

# Command to run on container start
CMD ["tail", "-f", "/dev/null"]
