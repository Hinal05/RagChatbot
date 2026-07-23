"""External action: live weather lookup via Open-Meteo (free, no API key).

This demonstrates the chatbot performing a real external action beyond
retrieval — the model decides *when* to call it (see chat.py routing step),
and the result is fed back into the model as fresh context.
"""
import requests

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

WMO_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "freezing fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    61: "light rain", 63: "moderate rain", 65: "heavy rain",
    71: "light snow", 73: "moderate snow", 75: "heavy snow",
    80: "light showers", 81: "moderate showers", 82: "violent showers",
    95: "thunderstorm",
}


def get_weather(location: str) -> dict:
    """Returns a small dict with current weather for `location`, or an error key."""
    try:
        geo = requests.get(GEOCODE_URL, params={"name": location, "count": 1}, timeout=8)
        geo.raise_for_status()
        results = geo.json().get("results")
        if not results:
            return {"error": f"Could not find a location matching '{location}'."}

        place = results[0]
        lat, lon = place["latitude"], place["longitude"]

        wx = requests.get(
            WEATHER_URL,
            params={"latitude": lat, "longitude": lon, "current_weather": True},
            timeout=8,
        )
        wx.raise_for_status()
        current = wx.json().get("current_weather", {})

        return {
            "location": f"{place['name']}, {place.get('country', '')}".strip(", "),
            "temperature_c": current.get("temperature"),
            "windspeed_kmh": current.get("windspeed"),
            "condition": WMO_CODES.get(current.get("weathercode"), "unknown"),
        }
    except requests.RequestException as exc:
        return {"error": f"Weather lookup failed: {exc}"}
