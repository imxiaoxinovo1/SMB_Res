@echo off
REM 运行随机森林模型 - 无数据泄露版本
REM 排除 AAR 和 ELA（这些是物质平衡的结果）

echo ======================================================================
echo 随机森林模型 - LOOCV（无数据泄露版本）
echo ======================================================================
echo.
echo 重要说明：
echo   - 已排除 AAR 和 ELA 特征
echo   - 这些特征是物质平衡的结果，会导致数据泄露
echo   - 仅使用物理驱动因素：气候、地理、冰川面积、时间
echo.
echo ======================================================================

REM 激活 conda 环境
call G:\Study\Anaconda\Scripts\activate
call conda activate smb

REM 运行模型
python RF_model_LOOCV_no_leakage.py

pause
