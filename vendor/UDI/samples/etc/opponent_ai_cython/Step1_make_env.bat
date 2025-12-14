mkdir C:\Workspace_OpponentAI
cd C:\Workspace_OpponentAI
py -3.12 -m venv .venv
call .venv\Scripts\activate.bat
pause
pip install C:\UDI\libs\ygo
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu129
pip install grpcio-tools==1.74.0
pip install pycryptodome==3.23.0
pip list
pause