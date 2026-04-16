@echo off
setlocal enabledelayedexpansion

:: 使用从 umat.bat 传递的项目源目录
if not defined PROJECT_SOURCE_DIR (
    echo 错误: PROJECT_SOURCE_DIR 未定义
    exit /b 1
)
set "OUTPUT_FILE=%PROJECT_SOURCE_DIR%\merge\source.F90"

:: 清空输出文件
echo. > "%OUTPUT_FILE%"
:: 添加源文件内容
@REM echo 添加 base_config.F90
@REM type "%PROJECT_SOURCE_DIR%\fortran\utils\include\base_config.F90" >> "%OUTPUT_FILE%"
@REM echo 添加 material_config.F90
@REM type "%PROJECT_SOURCE_DIR%\fortran\utils\include\material_config.F90" >> "%OUTPUT_FILE%"
@REM echo 添加 container.F90...
@REM type "%PROJECT_SOURCE_DIR%\fortran\update\include\container.F90" >> "%OUTPUT_FILE%"
@REM echo 添加 tensor_opt.F90...
@REM type  "%PROJECT_SOURCE_DIR%\fortran\utils\include\tensor_opt.F90"  >> "%OUTPUT_FILE%"
@REM echo 添加 preprocess.F90...
@REM type  "%PROJECT_SOURCE_DIR%\fortran\utils\include\preprocess.F90"  >> "%OUTPUT_FILE%"
@REM echo 添加 exception.F90...
@REM type  "%PROJECT_SOURCE_DIR%\fortran\utils\include\exception.F90"  >> "%OUTPUT_FILE%"
@REM echo 添加 elastic.F90...
@REM type  "%PROJECT_SOURCE_DIR%\fortran\update\include\elastic.F90"  >> "%OUTPUT_FILE%"
@REM echo 添加 plastic.F90...
@REM type  "%PROJECT_SOURCE_DIR%\fortran\update\include\plastic.F90"  >> "%OUTPUT_FILE%"
@REM echo 添加 math.F90...
@REM type "%PROJECT_SOURCE_DIR%\fortran\update\include\math.F90"  >> "%OUTPUT_FILE%"
:: 添加 impl
@REM echo 添加 container_impl.F90...
@REM type "%PROJECT_SOURCE_DIR%\fortran\update\src\container_impl.F90" >> "%OUTPUT_FILE%"
@REM echo 添加 exception_impl.F90...
@REM type "%PROJECT_SOURCE_DIR%\fortran\utils\src\exception_impl.F90" >> "%OUTPUT_FILE%"
@REM echo 添加 preprocess_impl.F90...
@REM type "%PROJECT_SOURCE_DIR%\fortran\utils\src\preprocess_impl.F90" >> "%OUTPUT_FILE%"
@REM echo 添加 tensor_opt_impl.F90...
@REM type  "%PROJECT_SOURCE_DIR%\fortran\utils\src\tensor_opt_impl.F90"  >> "%OUTPUT_FILE%"
@REM echo 添加 elastic_impl.F90...
@REM type  "%PROJECT_SOURCE_DIR%\fortran\update\src\elastic_impl.F90"  >> "%OUTPUT_FILE%"
@REM echo 添加 plastic_impl.F90...
@REM type  "%PROJECT_SOURCE_DIR%\fortran\update\src\plastic_impl.F90"  >> "%OUTPUT_FILE%"
@REM echo 添加 math_impl.F90...
@REM type "%PROJECT_SOURCE_DIR%\fortran\update\src\math_impl.F90"  >> "%OUTPUT_FILE%"
:: 添加源文件内容
@REM echo add umat_fortran.F90
@REM type "%PROJECT_SOURCE_DIR%\fortran\interface\umat_fortran.F90" >> "%OUTPUT_FILE%"
@REM echo add interface.F90
@REM type "%PROJECT_SOURCE_DIR%\fortran\interface.f90" >> "%OUTPUT_FILE%"
@REM echo add umat.F90
@REM type "%PROJECT_SOURCE_DIR%\fortran\umat.f90" >> "%OUTPUT_FILE%"
echo.
echo cpp源文件合并完成
echo output dir: %OUTPUT_FILE%

endlocal
