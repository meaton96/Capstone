FROM python:3.10.12-slim

# Install system dependencies including OpenSSH client and Git
RUN apt-get update && apt-get install -y \
    openssh-client \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /code

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose ports for TensorBoard (6006) and ml-agents (5004)
EXPOSE 6006 5004

# Keeps the container alive in the background so you can attach to it
CMD ["tail", "-f", "/dev/null"]