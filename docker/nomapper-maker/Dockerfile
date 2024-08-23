FROM ubuntu:focal

# metadata
LABEL software="noMapper Maker"
LABEL software.description="Maker of a selector separating reads that do not belong to the gene being searched from those that may belong to it."
LABEL author="Tomasz Strzoda"
LABEL author.email="tomasz.strzoda@polsl.pl"

# repo for samtools
RUN echo "deb http://archive.ubuntu.com/ubuntu focal universe" >> /etc/apt/sources.list

# aliases
COPY ./aliases.sh /aliases.sh
RUN cat /aliases.sh >> /root/.bashrc

# essentials
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Warsaw
RUN apt-get update
RUN apt-get install -y curl bzip2 python3 python3-distutils python3-pip python3-apt samtools
RUN apt-get autoclean && rm -rf /var/lib/apt/lists/*

# minimap2
WORKDIR /tool
ARG MINIMAP2_VERSION="2.24"
RUN curl -L https://github.com/lh3/minimap2/releases/download/v${MINIMAP2_VERSION}/minimap2-${MINIMAP2_VERSION}_x64-linux.tar.bz2 | tar -jxvf -
ENV PATH="${PATH}:/tool/minimap2-${MINIMAP2_VERSION}_x64-linux"

# porechop
WORKDIR /tool
ARG PORECHOP_VERSION="0.2.4"
RUN curl -L https://github.com/rrwick/Porechop/archive/refs/tags/v${PORECHOP_VERSION}.tar.gz | tar -zxvf -
WORKDIR /tool/Porechop-${PORECHOP_VERSION}
RUN python3 setup.py install

# filtlong
WORKDIR /tool
ARG FILTLONG_VERSION="0.2.1"
RUN curl -L https://github.com/rrwick/Filtlong/archive/refs/tags/v${FILTLONG_VERSION}.tar.gz | tar -zxvf -
WORKDIR /tool/Filtlong-${FILTLONG_VERSION}
RUN make -j
ENV PATH="${PATH}:/tool/Filtlong-${FILTLONG_VERSION}/bin"

# scripts
WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
RUN pip3 install -r requirements.txt
COPY ./app /app

CMD ["bash"]