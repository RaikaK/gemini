cd C:\Workspace_OpponentAI
call .venv\Scripts\activate.bat 
cd C:\UDI\samples\etc\opponent_ai_cython\ClientCy\
python run.py --tcphost %1 --tcpport 50002 --LoopNum 100000 --GrpcAddressPort localhost:50806 --RandomActionRate 0.0