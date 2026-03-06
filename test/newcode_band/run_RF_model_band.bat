@echo off
REM 运行高程带数据随机森林模型

echo ======================================================================
echo 高程带数据 - 随机森林模型（留一年交叉验证）
echo ======================================================================
echo.
echo 数据特点：
echo   - 高程带分层数据（每个冰川-年有多个高程带）
echo   - 包含高程特征（ELEVATION_MIDPOINT, ELEVATION_NORMALIZED 等）
echo   - 不包含 AAR 和 ELA（避免数据泄露）
echo.
echo 模型配置：
echo   - RFE 特征选择（寻找 R^2 最高的特征子集）
echo   - 留一年交叉验证
echo   - 随机森林回归
echo.
echo ======================================================================

REM 激活 conda 环境
call G:\Study\Anaconda\Scripts\activate
call conda activate smb

REM 运行模型
python RF_model_LOOCV_band.py

pause
