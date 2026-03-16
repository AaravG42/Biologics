import requests
import os

url = "https://opig.stats.ox.ac.uk/webapps/opig/sabdab-therabdab/data/TheraSAbDab_SeqStruc_Data_Download.csv"
output_path = "data/TheraSAbDab_SeqStruc_Data_Download.csv"

def download_data():
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    
    print(f"Downloading data from {url}...")
    try:
        response = requests.get(url, headers=headers, allow_redirects=True, timeout=30)
        response.raise_for_status()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(response.content)
            
        print(f"Successfully downloaded {len(response.content)} bytes to {output_path}")
        
    except Exception as e:
        print(f"Failed to download data: {e}")

if __name__ == "__main__":
    download_data()
