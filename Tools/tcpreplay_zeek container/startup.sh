#!/bin/bash

echo "Starting Zeek Traffic Analyzer..."

# dummy interface that will be listened by zeek
if ip link add dummy0 type dummy 2>/dev/null; then
    ip addr add 10.0.0.1/24 dev dummy0
    ip link set dummy0 up
    echo "Created dummy0 interface"
    INTERFACE="dummy0"
    USE_INTERFACE=true
else
    echo "Cannot create dummy interface, will use file analysis mode"
    USE_INTERFACE=false
fi

# Start Zeek 
if [ "$USE_INTERFACE" = true ]; then
    cd /zeek-logs
    zeek -i $INTERFACE &
    ZEEK_PID=$!
    echo "Zeek started with PID: $ZEEK_PID"
fi

# play the pcaps
process_pcap() {
    local pcap_file=$1
    echo "Processing: $pcap_file"
    
    if [ "$USE_INTERFACE" = true ]; then
        # replay through the dummy interface
        tcpreplay --intf1=$INTERFACE "$pcap_file" 2>/dev/null
        if [ $? -ne 0 ]; then
            echo "Tcpreplay failed, falling back to direct file analysis"
            cd /zeek-logs
            zeek -r "$pcap_file"
        fi
    else
        # Direct file analysis
        cd /zeek-logs
        zeek -r "$pcap_file"
    fi
    
    echo "Finished processing: $pcap_file"
}

# scan for pcaps file within the specified folder
for pcap in /pcap/*.pcap /pcap/*.pcapng; do
    if [ -f "$pcap" ]; then
        process_pcap "$pcap"
    fi
done

# check for new PCAP files
echo "Monitoring /pcap directory for new files..."
inotifywait -m /pcap -e create -e moved_to -e close_write |
while read dir action file; do
    if [[ "$file" =~ \.(pcap|pcapng)$ ]]; then
        echo "New file detected: $file"
        sleep 1  
        process_pcap "/pcap/$file"
    fi
done