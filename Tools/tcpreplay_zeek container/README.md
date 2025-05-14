# Zeek Replayer Docker Setup Guide

## 1. Pull the Image

```bash
docker pull tkhoi/zeek-replayer:latest
```

## 2. Create Folders for PCAP Files and Logs

You can change the path as you wish. Then paste the PCAP files into the `pcaps` folder after creating it.

```bash
mkdir C:\zeek-analysis\pcaps
mkdir C:\zeek-analysis\logs
```

## 3. Run the Container

```bash
docker run -it --rm --name zeek-replayer --privileged -v C:\zeek-analysis\pcaps:/pcap:ro -v C:\zeek-analysis\logs:/zeek-logs:rw tkhoi/zeek-replayer:latest
```