FROM python:3.11-slim

WORKDIR /app

# Copy and install dependencies
COPY smc_ultra_v2/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY smc_ultra_v2/ .

# Create data directories
RUN mkdir -p data/historical models backtest_results

# Start bot
CMD ["python", "run_bot.py"]
