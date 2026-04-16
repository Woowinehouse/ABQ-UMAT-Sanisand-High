@echo off
setlocal enabledelayedexpansion
rem --- 设置UTF-8输出编码 ---
powershell -Command "$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)"

rem --- 显示当前终端路径 ---
echo ======================================================================
echo   当前终端路径信息
echo ----------------------------------------------------------------------
echo 脚本启动时的当前工作目录: %cd%
echo ======================================================================
echo.

rem --- 1) 启动 VS2022 vcvars，以 v140 模拟 VC++14.0 ---
@REM call  "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" x86_amd64
call  "E:\VS2026\VC\Auxiliary\Build\vcvars64.bat" x64
rem --- 2) 启动 Intel 环境（如果需要） ---
call "E:\oneapi2025\compiler\latest\env\vars.bat" intel64 vs2026
@REM call "E:\IntelSWTools\compilers_and_libraries_2016.1.146\windows\bin\ipsxe-comp-vars.bat" intel64 vs2015

rem --- 添加运行时库兼容性修复 ---
set "VSCMD_ARG_HOST_ARCH=x64"
set "VSCMD_ARG_TGT_ARCH=x64"
set "Platform=x64"
rem --- 3) 把项目 include/lib 路径放到环境变量前面，确保编译器能找到你的头与兼容的 VC/SDK 头 ---
rem 自动获取项目根目录（umat.bat所在目录的父目录）
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "PROJECT_SOURCE_DIR=%SCRIPT_DIR%\.."
rem 保存当前终端路径（运行批处理文件时的原始工作目录）
set "ORIGINAL_WORKING_DIR=%cd%"
cd /d "%PROJECT_SOURCE_DIR%"
set "PROJECT_SOURCE_DIR=%cd%"
rem -- 头文件路径
set "PROJECT_INCLUDE=%PROJECT_SOURCE_DIR%\build-vs2026\modules\Debug"
rem -- 项目lib路径
set "PROJECT_LIB=%PROJECT_SOURCE_DIR%\build-vs2026\lib\Debug"
rem ---添加项目ddl到PATH ---
set "PATH=%PROJECT_LIB%;%PATH%"
rem --- 第三方
set "LIBTORCH_INCLUDE=%PROJECT_SOURCE_DIR%\third_party\libtorch_cpu_2_11_0\include"
set "LIBTORCH_INCLUDE_API=%LIBTORCH_INCLUDE%\torch\csrc\api\include"
set "LIBTORCH_LIB=%PROJECT_SOURCE_DIR%\third_party\libtorch_cpu_2_11_0\lib"
rem --- 添加LibTorch DLL路径到PATH ---
set "PATH=%LIBTORCH_LIB%;%PATH%"
echo 已将LibTorch DLL路径添加到PATH: %LIBTORCH_LIB%
rem --- detect latest Windows Kits 10 include version and set include/lib paths ---
set "WINDOWS_VC_INCLUDE=E:\vs2026\VC\Tools\MSVC\14.50.35717\include"
set "WINDOWS_VC_LIB=E:\vs2026\VC\Tools\MSVC\14.50.35717\lib\x64"
set "ONEAPI_LIB=E:\oneapi2022\compiler\latest\windows\compiler\lib\intel64_win"
echo ======================================================================
echo   环境路径信息
echo ----------------------------------------------------------------------
echo.
echo [当前终端路径]
echo ORIGINAL_WORKING_DIR: %ORIGINAL_WORKING_DIR%
echo.
echo [头文件路径]
echo PROJECT_INCLUDE: %PROJECT_INCLUDE%
echo LIBTORCH_INCLUDE: %LIBTORCH_INCLUDE%
echo LIBTORCH_INCLUDE_API: %LIBTORCH_INCLUDE_API%
echo WINDOWS_VC_INCLUDE: %WINDOWS_VC_INCLUDE%
echo.
echo [动态库路径]
echo PROJECT_LIB: %PROJECT_LIB%
echo WINDOWS_VC_LIB: %WINDOWS_VC_LIB%
echo ONEAPI_LIB: %ONEAPI_LIB%
echo LIBTORCH_LIB: %LIBTORCH_LIB%
echo.
echo [Windows SDK 检测]
set "WIN_KIT_VER="
for /f "delims=" %%V in ('dir /b /ad "C:\Program Files (x86)\Windows Kits\10\Include" 2^>nul ^| sort /r') do (
	set "WIN_KIT_VER=10.0.26100.0"
	goto :FoundWinKit
)
:FoundWinKit
if defined WIN_KIT_VER (
	echo WIN_KIT_VER: %WIN_KIT_VER%

	set "WIN_KIT_UCRT=C:\Program Files (x86)\Windows Kits\10\Include\%WIN_KIT_VER%\ucrt"
	set "WIN_KIT_SHARED=C:\Program Files (x86)\Windows Kits\10\Include\%WIN_KIT_VER%\shared"
	set "WIN_KIT_UM=C:\Program Files (x86)\Windows Kits\10\Include\%WIN_KIT_VER%\um"
	set "WIN_KIT_LIB_UCRT=C:\Program Files (x86)\Windows Kits\10\Lib\%WIN_KIT_VER%\ucrt\x64"
	set "WIN_KIT_LIB_UM=C:\Program Files (x86)\Windows Kits\10\Lib\%WIN_KIT_VER%\um\x64"

	echo WIN_KIT_UCRT: !WIN_KIT_UCRT!
	echo WIN_KIT_SHARED: !WIN_KIT_SHARED!
	echo WIN_KIT_UM: !WIN_KIT_UM!
	echo WIN_KIT_LIB_UCRT: !WIN_KIT_LIB_UCRT!
	echo WIN_KIT_LIB_UM: !WIN_KIT_LIB_UM!

	rem 构建INCLUDE路径 - 避免多余分号
	set "INCLUDE=!PROJECT_INCLUDE!"
  rem --- 添加第三方库
  set "INCLUDE=!INCLUDE!;!LIBTORCH_INCLUDE!"
  set "INCLUDE=!INCLUDE!;!LIBTORCH_INCLUDE_API!"
	set "INCLUDE=!INCLUDE!;!WIN_KIT_UCRT!"
	set "INCLUDE=!INCLUDE!;!WIN_KIT_SHARED!"
	set "INCLUDE=!INCLUDE!;!WIN_KIT_UM!"
	set "INCLUDE=!INCLUDE!;!WINDOWS_VC_INCLUDE!"
	rem 构建LIB路径 - 避免多余分号
	set "LIB=!PROJECT_LIB!"
  set "LIB=!LIB!;!LIBTORCH_LIB!"
	set "LIB=!LIB!;!WIN_KIT_LIB_UCRT!"
	set "LIB=!LIB!;!WIN_KIT_LIB_UM!"
	set "LIB=!LIB!;!WINDOWS_VC_LIB!"
  set "LIB=!LIB!;!ONEAPI_LIB!"
) else (
	echo [回退路径 - 未找到 Windows SDK]
	set "INCLUDE=%PROJECT_SOURCE_DIR%\include;%PROJECT_SOURCE_DIR%\include\utils;C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\include"
	set "LIB=%PROJECT_SOURCE_DIR%\lib;C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\lib\amd64"
)
echo.
echo [最终环境变量]
echo INCLUDE: %INCLUDE%
echo LIB: %LIB%

echo.
echo [路径验证]
echo 检查 ucrt 路径是否存在:
if exist "%WIN_KIT_UCRT%" (echo   ✓ 存在) else (echo   ✗ 不存在)
echo 检查 shared 路径是否存在:
if exist "%WIN_KIT_SHARED%" (echo   ✓ 存在) else (echo   ✗ 不存在)
echo 检查 um 路径是否存在:
if exist "%WIN_KIT_UM%" (echo   ✓ 存在) else (echo   ✗ 不存在)
echo 检查 VC include 路径是否存在:
if exist "E:\vs2022\VC\Tools\MSVC\14.50.35717\include" (echo   ✓ 存在) else (echo   ✗ 不存在)

rem --- 4) 合并所有源文件 ---
echo.
echo ======================================================================
echo   合并源文件
echo ----------------------------------------------------------------------
call "%PROJECT_SOURCE_DIR%\scripts\merge_cpp.bat"

rem --- 5) 删除旧的编译文件 ---
echo.
echo ======================================================================
echo   清理旧的编译文件
echo ----------------------------------------------------------------------
if exist "%PROJECT_SOURCE_DIR%\source-std.obj" (
    echo 删除 source-std.obj...
    del "%PROJECT_SOURCE_DIR%\source-std.obj"
) else (
    echo source-std.obj 不存在，无需删除
)

if exist "%PROJECT_SOURCE_DIR%\standardU.dll" (
    echo 删除 standardU.dll...
    del "%PROJECT_SOURCE_DIR%\standardU.dll"
) else (
    echo standardU.dll 不存在，无需删除
)
echo ======================================================================
@REM rem --- 6) 调用 Abaqus 的 SMALauncher（使用合并后的文件） ---
@REM rem If no arguments are provided, this wrapper was likely invoked directly.
@REM rem Use a label-based check to avoid batch parser issues with parenthesis in environment variables.
if "%~1"=="" goto :NoArgs


rem --- 6) 调用 Abaqus 的 SMALauncher（使用合并后的文件） ---
rem If no arguments are provided, this wrapper was likely invoked directly.
rem Use a label-based check to avoid batch parser issues with parenthesis in environment variables.
if "%~1"=="" goto :NoArgs

"E:\ABAQUS2025\product\win_b64\code\bin\SMALauncher.exe" %*
goto :EOF

:NoArgs
echo No arguments passed to umat.bat. This wrapper should be invoked by the Abaqus launcher (abq2025).
echo Example: abq2025 make library=umat/umat_merged.cpp
exit /b 0
