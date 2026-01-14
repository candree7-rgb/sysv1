FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Create data directories
RUN mkdir -p data/historical models backtest_results

# Start script that downloads data first, then runs bot
CMD ["python", "run_bot.py"]
