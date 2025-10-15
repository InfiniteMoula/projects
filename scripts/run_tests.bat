@echo off
setlocal enabledelayedexpansion

pytest -q
if errorlevel 1 exit /b 1

ruff check .
if errorlevel 1 exit /b 1

mypy --ignore-missing-imports
