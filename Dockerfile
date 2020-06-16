FROM ubuntu:20.04

# Update and upgrade distribution
RUN apt-get update && \
    apt-get upgrade -y

# Tools necessary for installing and configuring Ubuntu
RUN apt-get install -y \
    apt-utils \
    locales \
    tzdata

# Timezone
RUN echo "Europe/Paris" | tee /etc/timezone && \
    ln -fs /usr/share/zoneinfo/Europe/Paris /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata

# Locale with UTF-8 support
RUN echo en_US.UTF-8 UTF-8 >> /etc/locale.gen && \
    locale-gen && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Required for Gudhi compilation
RUN apt-get install -y texlive texlive-latex-extra \
    locales \
    python3 \
    python3-pip \
    python3-tk

# apt clean up
RUN apt autoremove && rm -rf /var/lib/apt/lists/*

ADD .binder/requirements.txt /

RUN pip3 install -r requirements.txt
RUN pip3 install jupyter
