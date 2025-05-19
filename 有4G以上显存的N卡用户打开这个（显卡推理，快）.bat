chcp 65001
@echo off
echo 正在以显卡加速模式启动中，请稍后……
set WEBUI_CONFIG_DEVICE=cuda
.\bertvenv\python.exe webui.py
pause