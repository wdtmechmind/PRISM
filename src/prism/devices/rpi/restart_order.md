先开ups，过1min再插usb

一键命令（默认参数）：

```bash
bash scripts/restart_rpi_usb.sh
```

带参数版本：

```bash
bash scripts/restart_rpi_usb.sh enx3ed0bbcddb23 10.12.194.1 xining rpi-usb-host
```

参数顺序：

1. IF_NAME（默认 enx3ed0bbcddb23）
2. RPI_IP（默认 10.12.194.1）
3. SSH_USER（默认 xining）
4. CONNECTION_NAME（默认 rpi-usb-host）

可选：如果只想配置网络不自动 ssh，运行前设置环境变量：

```bash
AUTO_SSH=0 bash scripts/restart_rpi_usb.sh
```