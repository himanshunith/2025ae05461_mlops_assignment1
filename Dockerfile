# Use a lightweight official Python 3.10 image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy dependency file and install first (leverages Docker cache for faster rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application scripts
COPY data_pipeline.py model_pipeline.py inference.py app.py ./

# Copy the serialized model artifact
# Note: We must recreate the directory structure expected by inference.py
COPY heart_disease_dataset/model_artifacts/best_model.pkl ./heart_disease_dataset/model_artifacts/

# Expose the port FastAPI will run on
EXPOSE 8000

# Start the uvicorn server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]