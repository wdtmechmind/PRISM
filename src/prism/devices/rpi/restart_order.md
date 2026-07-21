先开ups，过1min再插usb



IF=enx3ed0bbcddb23
if [ -n "$IF" ]; then
    sudo nmcli device set "$IF" managed yes
    sudo nmcli connection modify rpi-usb-host connection.interface-name "$IF"
    sudo nmcli connection up rpi-usb-host
    ping -c 3 10.12.194.1
    ssh xining@10.12.194.1
else
    echo "未检测到 Raspberry Pi USB 网卡，请重新插拔 USB 数据线"
fi