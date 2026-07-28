#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/restart_rpi_usb.sh [IF_NAME] [RPI_IP] [SSH_USER] [CONNECTION_NAME]
# Example:
#   bash scripts/restart_rpi_usb.sh enx3ed0bbcddb23 10.12.194.1 xining rpi-usb-host

IF_NAME="${1:-enx3ed0bbcddb23}"
RPI_IP="${2:-10.12.194.1}"
SSH_USER="${3:-xining}"
CONNECTION_NAME="${4:-rpi-usb-host}"
AUTO_SSH="${AUTO_SSH:-1}"

if ! ip link show "$IF_NAME" >/dev/null 2>&1; then
    echo "未检测到 Raspberry Pi USB 网卡: $IF_NAME，请重新插拔 USB 数据线"
    exit 1
fi

echo "[1/4] 设置网卡受 NetworkManager 管理: $IF_NAME"
sudo nmcli device set "$IF_NAME" managed yes

echo "[2/4] 绑定连接配置: $CONNECTION_NAME -> $IF_NAME"
sudo nmcli connection modify "$CONNECTION_NAME" connection.interface-name "$IF_NAME"

echo "[3/4] 激活连接: $CONNECTION_NAME"
sudo nmcli connection up "$CONNECTION_NAME"

echo "[4/4] Ping 测试: $RPI_IP"
ping -c 3 "$RPI_IP"

if [[ "$AUTO_SSH" == "1" ]]; then
    echo "连接 SSH: ${SSH_USER}@${RPI_IP}"
    exec ssh "${SSH_USER}@${RPI_IP}"
fi

echo "网络已就绪（AUTO_SSH=$AUTO_SSH，跳过 SSH 登录）"