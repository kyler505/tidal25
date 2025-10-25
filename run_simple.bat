@echo off
cd /d "%~dp0"
set PYTHONPATH=%CD%
streamlit run src\demo_app_enhanced.py

