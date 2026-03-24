FROM python:3.10.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    openssh-client \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy only the configuration files first to leverage Docker cache
COPY pyproject.toml .

# Install the project and its dependencies 
# The "-e ." installs the current directory in 'editable' mode
RUN pip install --no-cache-dir -e .

# Expose ports for TensorBoard (6006) and ml-agents (5004)
EXPOSE 6006 5004

CMD ["tail", "-f", "/dev/null"]