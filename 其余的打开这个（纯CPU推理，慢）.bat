chcp 65001
@echo off
echo 正在以纯cpu模式启动中，请稍后……
set WEBUI_CONFIG_DEVICE=cpu
.\bertvenv\python.exe webui.py
pause