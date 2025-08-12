# How to use

## 1. Install dependencies

```bash
uv pip install flask flask-cors sentencepiece protobuf
```

## 2. SSH connection

Please connect to the server using SSH with port forwarding. This allows you to access the server's local port (5000) from your local machine. If you are using a different port, please change the port number accordingly.

The bottom line of "app.py" shows the port number that the server is running on. By default, it is set to 5000.
If you are using a different port, please change the port number accordingly.

```bash
ssh -L 5000:localhost:5000 f2530116@gpu01.ced.cei.uec.ac.jp
```

## 3. Run the server

```bash
# 必要であれば
source .venv/bin/activate
```

```bash
python app.py
```

