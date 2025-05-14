# Pull the image
docker pull tkhoi/zeek-replayer:latest

# Create folders for PCAP files and logs
# You can change the path
# Patse the pcap files into the pcaps folder after creating this folder
mkdir C:\zeek-analysis\pcaps
mkdir C:\zeek-analysis\logs

# Run the container
docker run -it --rm ^
    --name zeek-replayer ^
    --privileged ^
    -v C:\zeek-analysis\pcaps:/pcap:ro ^
    -v C:\zeek-analysis\logs:/zeek-logs:rw ^
    tkhoi/zeek-replayer:latest