# TinyBeats Backend

This repository contains the backend code for TinyBeats, a file upload and processing service built using Flask and Celery. TinyBeats allows users to upload files in base64-encoded format and then calculates embeddings for text-based files, storing the results in a Supabase database.

## Getting Started

To run the TinyBeats backend, follow these instructions:

### Prerequisites

- Python 3.6 or higher
- Redis server (for Celery)

### Installation

1. Clone this GitHub repository to your local machine.

```bash
git clone https://github.com/your_username/TinyBeats.git
```

2. Change to the project's directory.

```bash
cd TinyBeats
```

3. Create Virtual env in directory and activate it.

```bash
python3 -m venv venv
souce venv/bin/activate
```
Note: If you haven't installed python3.10-venv you can install sudo apt install python3.10-venv then run above command, also 3.10 is my version of python you check yours and put it there, like for 3.8, python3.8-venv.

4. Install the required Python packages.

```bash
pip install -r requirements.txt
```

5. Set up the environment variables.

Before running the application, ensure that you have set the following environment variables:

- `BUCKET_NAME`: The name of the bucket in your Supabase storage where files will be uploaded and downloaded.

Here are the general steps to install Redis and Celery on a Linux system using the package manager:

1. Install Redis:
   Redis is usually available through the package manager of most Linux distributions. You can install it by running the following commands based on your package manager:

   For Debian/Ubuntu systems:
   ```bash
   sudo apt update
   sudo apt install redis-server
   ```

   For Red Hat/Fedora systems:
   ```bash
   sudo dnf install redis
   ```

   For CentOS systems:
   ```bash
   sudo yum install redis
   ```

2. Install Celery:
   Celery is a Python library, so it's typically installed using Python's package manager. If you don't have already, you can install them with:

   For Debian/Ubuntu systems:
   ```bash
   sudo apt update
   sudo apt install celery
   ```

3. Verify the installations:
   After completing the installation, you can check if Redis and Celery are installed correctly.

   For Redis, check the status of the Redis server:
   ```bash
   sudo systemctl status redis
   ```

   For Celery, you can check its version to verify the installation:
   ```bash
   celery --version
   ```

That's it! Now you should have Redis and Celery installed on your system. Remember that the actual commands may vary slightly depending on your Linux distribution, but the general approach remains the same.

### Running the Application

1. Start the Redis server (required for Celery).

```bash
redis-server
sudo systemctl start redis-server
sudo systemctl enable redis-server
suod systemctl status redis-server
```

2. Start the Celery worker.
   before starting you have to run celery, but for that you must activate your environment. 'source venv/bin/activate' like this.

you have to activate your virtual env before running celery
```bash
source venv/bin/activate
```

```bash
celery -A app.celery worker
```

3. Start the Flask application.

```bash
python app.py
```

The backend is now up and running on `http://localhost:5000`.

## API Endpoints

The TinyBeats backend provides the following API endpoint:

### 1. `/upload` (GET/POST)

This endpoint allows users to upload documents to the Supabase storage. The documents are processed to calculate embeddings in an asynchronous Celery task.

#### Request

- Method: POST
- Content-Type: application/json

**Body:**

```json
{
  "file_name": "example.pdf",
  "file_content": "base64 encoded string",
  "category_index": "index of the file"
}
```

#### Response

- Success: 201 Created
- Failure: 400 Bad Request

### 2. `/chat` (GET/POST)

This endpoint allows users to chat with the backend using a query string. The backend retrieves the most relevant documents based on the query using OpenAI embeddings.

#### Request

- Method: POST
- Content-Type: application/json

**Body:**

```json
{
  "query": "query string",
  "category_index": "index of the file"
}
```

#### Response

- Success: 200 OK
- Failure: 400 Bad Request

### 3. `/delete_category` (GET/POST)

This endpoint allows users to delete all documents belonging to a specific category index from the Supabase storage.

#### Request

- Method: POST
- Content-Type: application/json

**Body:**

```json
{
  "category_index": "index of the file"
}
```

#### Response

- Success: 200 OK
- Failure: 400 Bad Request

### 4. `/delete_file` (GET/POST)

This endpoint allows users to delete a specific document by providing its filename.

#### Request

- Method: POST
- Content-Type: application/json

**Body:**

```json
{
  "file_name": "name of the file"
}
```

#### Response

- Success: 200 OK
- Failure: 400 Bad Request

## Important Note

Ensure that you have set up the Supabase client properly, including the required credentials, to enable the storage and retrieval of files and embeddings.

For more information on using TinyBeats, please refer to the API documentation.

## Contributing

If you wish to contribute to this project, please fork the repository and submit a pull request with your proposed changes. We appreciate your contributions!

## License

This project is licensed under a commercial license. All rights reserved. Do not copy or distribute without writter permission by Nikolai Manek or TerraMD LLC.

## Acknowledgments

Thank you to all the contributors and open-source projects that made this project possible. Your support is greatly appreciated.

---

This README file provides a basic overview of the TinyBeats backend, including setup instructions, API documentation, and important details about the background task for calculating embeddings. You can customize this README with additional information as needed for your specific project.
