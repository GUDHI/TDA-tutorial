FROM ubuntu:20.04

ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Update and upgrade distribution
RUN apt-get update && apt-get install -y \
      texlive texlive-latex-extra \
      python3 \
      python3-pip \
      python3-tk \
    && rm -rf /var/lib/apt/lists/*
