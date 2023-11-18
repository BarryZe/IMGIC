import requests
import os
from dotenv import load_dotenv

load_dotenv("client_info.env")
client_id = os.getenv('client_id')
client_secret = os.getenv('client_secret')

# Get an access token from Spotify
def get_token(client_id, client_secret):
    # Define endpoint
    endpoint = 'https://accounts.spotify.com/api/token'

    # Define parameters
    params = {"grant_type": "client_credentials", "client_id": client_id, "client_secret": client_secret}

    # Define header
    header = {"Content-Type": "application/x-www-form-urlencoded"}


    # Send request
    response = requests.post(endpoint, headers=header, data=params)
    print(response.text)
    # Convert response to JSON
    response_json = response.json()

    # Extract access token
    access_token = response_json['access_token']

    return access_token

def get_song_list(access_token, search_term):
    # Define endpoint
    endpoint = 'https://api.spotify.com/v1/search'

    # Define parameters
    params = {"q": search_term, "type": "track", "limit": "4"}

    # Define header
    header = {"Authorization": "Bearer " + access_token}

    # Send request
    response = requests.get(endpoint, headers=header, params=params)

    # Convert response to JSON
    response_json = response.json()

    # Extract items from JSON
    items = response_json['tracks']['items']

    # Extract song names from items
    song_names = [item['name'] for item in items]

    return song_names

access_token = get_token(client_id, client_secret)
print(get_song_list(access_token, "happy, upbeat, pop"))

