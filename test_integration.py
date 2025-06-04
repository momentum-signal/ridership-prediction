import requests
import json

# Test the complete integration
def test_integration():
    base_url = "http://127.0.0.1:5000"
    
    print("=== Testing Flask API Integration ===\n")
    
    # Test 1: Get stations
    print("1. Testing /stations endpoint...")
    try:
        response = requests.get(f"{base_url}/stations")
        if response.status_code == 200:
            stations_data = response.json()
            print(f"✅ Stations endpoint working - Found {len(stations_data['stations'])} stations")
            print(f"   Sample stations: {stations_data['stations'][:5]}")
        else:
            print(f"❌ Stations endpoint failed with status {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing stations endpoint: {e}")
    
    print()
    
    # Test 2: Make predictions with different scenarios
    test_cases = [
        {
            "name": "Monday (weekday)",
            "data": {"day_of_week": 1, "is_weekend": 0, "is_holiday": 0, "month": 6}
        },
        {
            "name": "Saturday (weekend)", 
            "data": {"day_of_week": 6, "is_weekend": 1, "is_holiday": 0, "month": 6}
        },
        {
            "name": "Sunday (weekend)",
            "data": {"day_of_week": 0, "is_weekend": 1, "is_holiday": 0, "month": 12}
        }
    ]
    
    print("2. Testing /predict endpoint...")
    for test_case in test_cases:
        try:
            response = requests.post(
                f"{base_url}/predict",
                headers={"Content-Type": "application/json"},
                data=json.dumps(test_case["data"])
            )
            
            if response.status_code == 200:
                prediction = response.json()
                print(f"✅ {test_case['name']}: {prediction['prediction']:.0f} passengers")
            else:
                print(f"❌ {test_case['name']}: Failed with status {response.status_code}")
                print(f"   Error: {response.text}")
        except Exception as e:
            print(f"❌ Error testing {test_case['name']}: {e}")
    
    print("\n=== Integration Test Complete ===")

if __name__ == "__main__":
    test_integration()
