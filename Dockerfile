# Use Ubuntu 20.04 as the parent image
FROM ubuntu:20.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Update package list and install required packages
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3 python3-pip nginx supervisor && \
    apt-get clean


# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app/

# Setup Nginx and Gunicorn
COPY nginx.conf /etc/nginx/sites-available/default
COPY supervisord.conf /etc/supervisor/conf.d/

COPY gunicorn.log /app/

RUN apt-get install -y redis-server

# Expose the Gunicorn port
EXPOSE 8080

RUN touch /app/gunicorn.log /app/gunicorn.err /app/celery.log /app/celery.err /app/redis.log /app/redis.err 

# Start Supervisor to manage Nginx and Gunicorn
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]

