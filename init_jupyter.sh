#!/bin/bash

# Activate the virtual environment
source /home/kevin/dev/tf217/tf217/bin/activate

# Run Jupyter Notebook and retrieve the token
jupyter-notebook --no-browser --ip=0.0.0.0 --NotebookApp.token='' &

# Wait for a moment to ensure Jupyter starts
sleep 5

# Retrieve the token from the Jupyter Notebook server logs
TOKEN=$(jupyter notebook list | grep -o 'token=[^&]*' | cut -d'=' -f2)

# Output the token
echo "Jupyter Notebook is running. Access it using the token: $TOKEN"