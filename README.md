# Azure AI Project

This project demonstrates how to set up and run an AI application using Azure services.

## Setup Instructions

### 1. Create a Virtual Environment

```bash
python3 -m venv ./venv
```

### 2. Create a .env File

Create a file named `.env` in the root directory of the project.

### 3. Configure Environment Variables

Add the following to your `.env` file:

```
AZURE_SUBSCRIPTION_ID=<Your Azure Subscription ID>
AZURE_REGION=<Your Azure Region>
```

Replace `<Your Azure Subscription ID>` and `<Your Azure Region>` with your actual Azure credentials.

### 4. Install Requirements

```bash
pip install -r requirements.txt
```

## Running the Project

1. Create a text document to serve as context for the AI application.

2. In `main.py`, use the `load_document()` function to load your document:

   ```python
   load_document('your_document.txt')
   ```

   Replace `'your_document.txt'` with the path to your context document.

3. Run the application:

   ```bash
   python3 main.py
   ```

## Additional Information

For more details about the project, its dependencies, or troubleshooting, please refer to the project documentation or contact the development team.


