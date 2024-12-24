import requests

# URL of the file to download
url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/dbpedia.bin"

# Local filename to save the downloaded file
local_filename = "dbpedia.bin"

try:
    # Send a GET request to the URL
    with requests.get(url, stream=True) as response:
        response.raise_for_status()  # Raise an error for HTTP issues
        # Write the content to a local file
        with open(local_filename, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):  # 8 KB chunks
                file.write(chunk)
    print(f"Downloaded successfully to {local_filename}")
except requests.exceptions.RequestException as e:
    print(f"Error downloading the file: {e}")

