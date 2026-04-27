import httpx
import asyncio
import json

TOMTOM_KEY = "po8RAcqnVRmWqMl9l4RRVp8IYZdv2P4X"

# Coordinates for a major intersection near DU (Mall Road area)
LAT = 28.6892
LON = 77.2106

async def fetch_live_traffic():
    # Updated URL for 'point' based flow data
    url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
    
    params = {
        "point": f"{LAT},{LON}",
        "unit": "KMPH",
        "key": TOMTOM_KEY
    }

    print(f" Pinging TomTom for traffic at North Campus ({LAT}, {LON})...")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                print("\n✅ Success! Real-time data received.")
                
                if 'flowSegmentData' in data:
                    flow = data['flowSegmentData']
                    print(f"\n Road: {flow.get('name', 'Main Road')}")
                    print(f" Current Speed: {flow.get('currentSpeed')} km/h")
                    print(f" Free Flow Speed: {flow.get('freeFlowSpeed')} km/h")
                    print(f"⏱  Travel Time: {flow.get('currentTravelTime')} seconds")
                else:
                    print(json.dumps(data, indent=2))
            else:
                print(f"\n Error {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(fetch_live_traffic())