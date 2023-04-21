#This code Creates a browse button which allows a user to open a file.

import tkinter as tk
from tkinter import filedialog
import requests
import boto3

# Define the AWS credentials
aws_access_key_id = 'your_access_key_id'
aws_secret_access_key = 'your_secret_access_key'

# Define the S3 bucket name and the file key (i.e., path)
bucket_name = 'inittowinithack'


s3_client = boto3.client('s3',
                         aws_access_key_id=aws_access_key_id,
                         aws_secret_access_key=aws_secret_access_key)

# Define the function that gets called when the Browse button is clicked
def browse_file():
    # Open a file dialog window and let the user select a file
    file_path = filedialog.askopenfilename()

    # If the user selected a file, upload it
    if file_path:
        upload_file(file_path)

# Define the function that uploads the file
def upload_file(file_path):
    url = 'https://example.com/upload'
    
    # Open the file in binary mode
    with open(file_path, 'rb') as f:
        # Create the files dictionary with the file object
        files = {'file': f}

        # Send the POST request with the files payload
        #response = requests.post(url, files=files) <--Old code
        
        # Upload the file to S3
        response = s3_client.upload_file(files, bucket_name, file_path)
    # Print the response text
    print(response.text)

# Create a GUI window with a Browse button
root = tk.Tk()
button = tk.Button(root, text='Browse', command=browse_file)
button.pack()

# Start the GUI event loop
root.mainloop()
