FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for FAISS and Streamlit
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the dependency file to the working directory
COPY requirements.txt .

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the content of the local directory to the working directory
COPY . .

# Download the model from Hugging Face since it's large and excluded from GitHub
RUN mkdir -p models && curl -sSL -O --output-dir models "https://huggingface.co/spaces/namansudo/Solarpower/resolve/main/models/solar_model.pkl"

# Expose port 8501 for Streamlit
EXPOSE 8501

# Run the Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
