FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Create data directories
RUN mkdir -p data/historical models backtest_results

# Default command (can be overridden)
CMD ["python", "main.py", "live", "--paper"]
