# Use an official Anaconda runtime as the base image
FROM python:3.11

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install the dependencies
RUN pip install -r requirements.txt

# Copy the application code to the working directory
COPY . .

# Expose the port on which the FastAPI application will run
EXPOSE 8000

# Run the FastAPI application
CMD ["python", "Fast_API.py"]
