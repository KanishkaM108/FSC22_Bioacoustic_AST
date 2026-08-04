@echo off
setlocal
echo Training FSC22 AST v2 seed 101...
python src\train_ast_v2_source_consistent.py --seed 101
if errorlevel 1 exit /b 1
echo Training FSC22 AST v2 seed 202...
python src\train_ast_v2_source_consistent.py --seed 202
if errorlevel 1 exit /b 1
echo Training FSC22 AST v2 seed 303...
python src\train_ast_v2_source_consistent.py --seed 303
if errorlevel 1 exit /b 1
echo Evaluating the three-seed AST v2 source-consistent ensemble...
python src\evaluate_source_consistent_ast_v2.py --seeds 101 202 303 --checkpoint-template "models/fsc22_ast_v2_seed{seed}.pt" --tag ast_v2
if errorlevel 1 exit /b 1
echo.
echo Finished. Results are in outputs\source_consistent_ast_ast_v2
endlocal
