@echo off

:: Check if ffmpeg is installed and in PATH
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo ffmpeg not found. Please install it and ensure it is in your PATH.
    exit /b 1
)

:: Check if a file was dragged
if "%~1"=="" (
    echo Please drag a video file onto this script.
    pause
    exit /b
)

:: Get the path and file name without extension
set "input_file=%~1"
set "input_dir=%~dp1"
set "input_name=%~n1"
set "output_file=%input_dir%%input_name%_combined.mp4"

:: Create a temporary file list in the same directory as the videos
set "file_list=%input_dir%file_list.txt"
del "%file_list%" >nul 2>&1

:: Generate the list of video files in the directory
for %%f in ("%input_dir%*.mp4") do (
    echo file '%%~f' >> "%file_list%"
)

:: Combine the videos using FFmpeg
ffmpeg -hide_banner -f concat -safe 0 -i "%file_list%" -c copy "%output_file%"

:: Cleanup
del "%file_list%"

echo-------------------------
echo.
echo Combined video created: %output_file%
timeout /t 5