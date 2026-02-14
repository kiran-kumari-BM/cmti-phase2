FROM python:3.9-slim

WORKDIR /app

# Install system dependencies required by OpenCV + Paddle
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy all project files
COPY . /app

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create folders
RUN mkdir -p uploads outputs

EXPOSE 5000

CMD ["python", "app.py"]
