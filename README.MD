# run the app

uvicorn app:app --host 0.0.0.0 --port 8000 --reload

ngrok http 8000

brew services start redis
# or: redis-server
