@echo off
setlocal
echo Running FSC22 label-free source-consistent AST evaluation...
python src\evaluate_source_consistent_ast_v2.py
if errorlevel 1 (
  echo.
  echo Evaluation failed. Copy the complete error and send it for review.
  exit /b 1
)
echo.
echo Finished. Results are in outputs\source_consistent_ast_legacy_three_seed
endlocal

