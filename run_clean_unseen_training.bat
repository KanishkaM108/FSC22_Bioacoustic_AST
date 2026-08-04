@echo off
setlocal
cd /d "%~dp0"

echo [1/5] Creating locked source-grouped split...
python src\prepare_clean_grouped_protocol.py
if errorlevel 1 goto :failed

echo [2/5] Training clean AST seed 101...
python src\train_ast_v2_source_consistent.py --seed 101 --epochs 40 --unfrozen-blocks 8 --tag clean_ast_v1
if errorlevel 1 goto :failed

echo [3/5] Training clean AST seed 202...
python src\train_ast_v2_source_consistent.py --seed 202 --epochs 40 --unfrozen-blocks 8 --tag clean_ast_v1
if errorlevel 1 goto :failed

echo [4/5] Training clean AST seed 303...
python src\train_ast_v2_source_consistent.py --seed 303 --epochs 40 --unfrozen-blocks 8 --tag clean_ast_v1
if errorlevel 1 goto :failed

echo [5/5] Evaluating the untouched test split once...
python src\evaluate_clean_ast_ensemble.py --tag clean_ast_v1
if errorlevel 1 goto :failed

echo.
echo CLEAN UNSEEN PIPELINE: PASSED
echo Upload the folder outputs\clean_ast_v1_locked_test for independent verification.
exit /b 0

:failed
echo.
echo PIPELINE STOPPED because a command failed. Do not continue to later stages.
exit /b 1
