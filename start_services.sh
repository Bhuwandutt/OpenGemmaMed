# 1. Start MongoDB (prompts for password if not running)
sudo systemctl start mongodb

# 1. Kill any existing zombies
echo "Cleaning GPU..."
nvidia-smi | grep 'python' | awk '{ print $5 }' | xargs -n1 kill -9


# 2. Start Backend in the background
echo "Starting Backend..."
cd ./backend && uvicorn main:app --port 8000 &

# 3. Start Frontend
echo "Starting Frontend..."
cd ./frontend && npm run dev
