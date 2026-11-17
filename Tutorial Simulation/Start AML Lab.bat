@echo off
rem Starts the tiny local web server (needs Node.js) and opens AML Lab in your browser.
rem No Node? Just double-click index.html instead - everything also works from file://
cd /d "%~dp0"
where node >nul 2>nul
if errorlevel 1 (
  echo Node.js not found - opening index.html directly.
  start "" "%~dp0index.html"
  exit /b
)
start "" http://localhost:5173
node serve.js 5173
