FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install prerequisites, esp the ip package
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    inotify-tools \
    tcpreplay \
    iproute2 \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

# Install Zeek
RUN echo 'deb http://download.opensuse.org/repositories/security:/zeek/xUbuntu_22.04/ /' > /etc/apt/sources.list.d/zeek.list && \
    wget -nv https://download.opensuse.org/repositories/security:zeek/xUbuntu_22.04/Release.key -O- | apt-key add - && \
    apt-get update && \
    apt-get install -y zeek && \
    rm -rf /var/lib/apt/lists/*


ENV PATH="/opt/zeek/bin:${PATH}"
RUN mkdir -p /pcap /zeek-logs
COPY startup.sh /startup.sh
RUN chmod +x /startup.sh

WORKDIR /

CMD ["/startup.sh"]