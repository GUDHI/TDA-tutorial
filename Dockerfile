# Inspired from https://jcrist.github.io/conda-docker-tips.html
FROM continuumio/miniconda3:4.7.12

ENV PYTHONDONTWRITEBYTECODE=true

RUN conda install --yes --freeze-installed -c conda-forge \
    nomkl \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn \
    tensorflow \
    gudhi \
    jupyter \
    git \
    zip \
    && conda install --yes --freeze-installed -c plotly plotly=4.5.0 \
    && conda clean -afy \
    && find /opt/conda/ -follow -type f -name '*.a' -delete      \
    && find /opt/conda/ -follow -type f -name '*.pyc' -delete    \
    && find /opt/conda/ -follow -type f -name '*.js.map' -delete
