# Use an official Python runtime
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy your project files into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for Flask or FastAPI
EXPOSE 8080

# Start the app
CMD ["python", "app.py"]

