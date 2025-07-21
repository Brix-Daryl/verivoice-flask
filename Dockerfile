FROM python:3.10-slim

# Install ffmpeg
RUN apt-get update && apt-get install -y ffmpeg

# Set the working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (Railway uses 5000 for Flask apps)
EXPOSE 5000

# Run your Flask app
CMD ["python", "app.py"]
