import requests
import os
import json
from PIL import Image
from io import BytesIO

def download_cricket_images(num_images=50, save_dir='cricket_dataset'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Modified API parameters for better results
    api_url = "https://api.inaturalist.org/v1/observations"
    params = {
        'taxon_name': 'Gryllidae',  # Cricket family
        'photos': True,
        'per_page': num_images,
        'order': 'desc',
        'quality_grade': 'any'  # Changed from 'research' to get more results
    }
    
    try:
        print("Connecting to iNaturalist API...")
        response = requests.get(api_url, params=params)
        
        # Print response status for debugging
        print(f"API Response Status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"Error: API returned status code {response.status_code}")
            print(f"Response content: {response.text}")
            return
            
        data = response.json()
        total_results = len(data.get('results', []))
        print(f"Found {total_results} cricket observations")
        
        if total_results == 0:
            # Try with broader search terms
            params['taxon_name'] = 'cricket'
            print("Retrying with broader search term 'cricket'...")
            response = requests.get(api_url, params=params)
            data = response.json()
            print(f"Found {len(data.get('results', []))} observations with broader search")
        
        for i, observation in enumerate(data.get('results', [])):
            if observation.get('photos') and len(observation['photos']) > 0:
                photo_url = observation['photos'][0]['url'].replace('square', 'medium')
                
                try:
                    img_response = requests.get(photo_url)
                    img = Image.open(BytesIO(img_response.content))
                    
                    filename = f"cricket_{i}.jpg"
                    img.save(os.path.join(save_dir, filename))
                    print(f"Downloaded {filename}")
                    
                except Exception as e:
                    print(f"Error downloading image {i}: {e}")

    except Exception as e:
        print(f"Error accessing API: {e}")
        print("Full error details:", str(e))

# Run the function
print("Starting cricket image download...")
download_cricket_images(50)
