from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pydantic import BaseModel
import math
import os

app = FastAPI(title="NASA Weather Advisor")

# السماح للفرونت إند بالاتصال
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# خدمة الملفات الثابتة للفرونت إند
app.mount("/static", StaticFiles(directory="static"), name="static")

class WeatherRequest(BaseModel):
    country: str
    city: str
    date: str
    user_type: str
    category: str

class LocationService:
    def get_coordinates(self, country: str, city: str) -> Dict:
        try:
            query = f"{city}, {country}"
            url = "https://nominatim.openstreetmap.org/search"
            params = {"q": query, "format": "json", "limit": 1}
            
            headers = {'User-Agent': 'WeatherApp/1.0'}
            response = requests.get(url, params=params, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data:
                    location = data[0]
                    return {
                        "success": True,
                        "lat": float(location["lat"]),
                        "lon": float(location["lon"]),
                        "display_name": location.get("display_name", f"{city}, {country}")
                    }
            return {"success": False, "error": "Location not found"}
        except Exception as e:
            return {"success": False, "error": f"Geocoding error: {str(e)}"}

class WeatherService:
    def __init__(self):
        self.nasa_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    
    def get_weather_data(self, lat: float, lon: float, target_date: str) -> Dict:
        try:
            target_dt = datetime.strptime(target_date, "%Y-%m-%d")
            current_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            
            if target_dt > current_date:
                historical_year = target_dt.year - 1
                start_date = f"{historical_year}{target_dt.strftime('%m%d')}"
            else:
                start_date = target_date.replace("-", "")
            
            end_date = start_date
            
            params = {
                "parameters": "T2M,RH2M,WS2M,PRECTOT",
                "start": start_date,
                "end": end_date,
                "latitude": lat,
                "longitude": lon,
                "community": "AG",
                "format": "JSON"
            }
            
            response = requests.get(self.nasa_url, params=params, timeout=15)
            
            if response.status_code == 200:
                return self._process_weather_data(response.json(), target_date)
            else:
                return self._get_default_weather(lat)
                
        except Exception as e:
            print(f"Weather API error: {e}")
            return self._get_default_weather(lat)
    
    def _process_weather_data(self, data: Dict, target_date: str) -> Dict:
        try:
            target_date_nasa = target_date.replace("-", "")
            
            temp_data = data['properties']['parameter']['T2M']
            rh_data = data['properties']['parameter']['RH2M']
            wind_data = data['properties']['parameter']['WS2M']
            rain_data = data['properties']['parameter']['PRECTOT']
            
            target_temp = temp_data.get(target_date_nasa, 20)
            target_rh = rh_data.get(target_date_nasa, 50)
            target_wind = wind_data.get(target_date_nasa, 3)
            target_rain = rain_data.get(target_date_nasa, 0)
            
            # تحديد حالة الطقس للفرونت إند
            weather_condition = self._get_weather_condition(target_temp, target_rain, target_rh)
            
            return {
                "temperature": round(target_temp, 1),
                "humidity": round(target_rh, 1),
                "wind_speed": round(target_wind, 1),
                "precipitation": round(target_rain, 1),
                "weather_condition": weather_condition,
                "success": True
            }
        except Exception as e:
            print(f"Weather processing error: {e}")
            return self._get_default_weather(0)
    
    def _get_weather_condition(self, temp: float, rain: float, humidity: float) -> str:
        """تحديد حالة الطقس للفرونت إند"""
        if rain > 10:
            return "rainy"
        elif rain > 0:
            return "cloudy"
        elif temp > 30:
            return "sunny"
        elif temp < 10:
            return "snowy"
        else:
            return "partly_cloudy"
    
    def _get_default_weather(self, lat: float) -> Dict:
        base_temp = 25 if abs(lat) < 30 else 15
        return {
            "temperature": base_temp,
            "humidity": 60.0,
            "wind_speed": 5.0,
            "precipitation": 0.0,
            "weather_condition": "partly_cloudy",
            "success": False,
            "note": "Using estimated weather data"
        }

class AdviceService:
    def __init__(self):
        self.advice_data = {
            "farmer": {
                "wheat": {
                    "rain_high": [
                        "Postpone irrigation today to avoid soil waterlogging",
                        "Cover the crop if possible to prevent damage",
                        "Check drainage channels to avoid water accumulation"
                    ],
                    "rain_low": [
                        "Reduce irrigation today as natural rainfall will help",
                        "Monitor soil moisture before adding extra water"
                    ],
                    "no_rain": [
                        "Irrigate normally according to your usual schedule",
                        "Check for early signs of water stress on leaves"
                    ]
                },
                "rice": {
                    "rain_high": [
                        "Ensure irrigation canals are open to prevent flooding",
                        "Check field boundaries around fields for leaks",
                        "Delay additional irrigation"
                    ],
                    "rain_low": [
                        "Light rain supports growth; reduce extra irrigation",
                        "Monitor water depth carefully in the paddy field"
                    ],
                    "no_rain": [
                        "Irrigate regularly to maintain 5-10 cm water depth",
                        "Watch for cracks in soil - early sign of dryness"
                    ]
                },
                "vegetables": {
                    "rain_high": [
                        "Cover crops with plastic sheets or greenhouse nets",
                        "Harvest ripe vegetables early to avoid spoilage",
                        "Improve drainage around rows"
                    ],
                    "rain_low": [
                        "Skip irrigation today; rainfall is sufficient",
                        "Remove excess weeds that trap moisture"
                    ],
                    "no_rain": [
                        "Irrigate in the morning or evening to reduce evaporation",
                        "Mulch soil to conserve water"
                    ]
                },
                "corn": {
                    "rain_high": [
                        "Inspect roots after heavy rain for damage",
                        "Support weak plants with soil mounding"
                    ],
                    "rain_low": [
                        "Reduce watering; rainfall is helpful",
                        "Check leaves for early fungal infections"
                    ],
                    "no_rain": [
                        "Irrigate regularly, especially during flowering stage",
                        "Apply fertilizer if leaves turn pale"
                    ]
                },
                "cotton": {
                    "rain_high": [
                        "Avoid irrigation during rainfall",
                        "Check for boll rot after heavy rain"
                    ],
                    "rain_low": [
                        "Rainfall supports growth; no extra irrigation needed",
                        "Monitor pest levels (aphids, whiteflies)"
                    ],
                    "no_rain": [
                        "Irrigate as scheduled; cotton requires consistent moisture",
                        "Inspect for leaf wilt in hot conditions"
                    ]
                }
            },
            "user": {
                "low": {
                    "rain_high": [
                        "Use plastic bags to cover shoes",
                        "Carry a small, inexpensive umbrella",
                        "Avoid drying clothes outdoors"
                    ],
                    "rain_low": [
                        "Wear a light jacket or hoodie",
                        "Carry a small plastic bag for essentials"
                    ],
                    "no_rain": [
                        "Wear regular clothes according to temperature",
                        "Stay hydrated if it's hot"
                    ]
                },
                "medium": {
                    "rain_high": [
                        "Wear a raincoat or water-resistant jacket",
                        "Carry a sturdy umbrella",
                        "Avoid slippery shoes; choose sneakers with grip"
                    ],
                    "rain_low": [
                        "A light waterproof jacket is enough",
                        "Keep a foldable umbrella in your bag"
                    ],
                    "no_rain": [
                        "Dress comfortably based on temperature",
                        "Plan outdoor activities freely"
                    ]
                },
                "high": {
                    "rain_high": [
                        "Wear waterproof jacket and waterproof shoes",
                        "Use anti-rain backpack cover",
                        "Consider driving instead of walking"
                    ],
                    "rain_low": [
                        "Stylish waterproof jacket recommended",
                        "Leather waterproof shoes are suitable"
                    ],
                    "no_rain": [
                        "Dress according to fashion and comfort",
                        "Sunglasses and light wear for sunny weather"
                    ]
                }
            }
        }
    
    def get_advice(self, user_type: str, category: str, precipitation: float, temperature: float) -> Dict:
        if precipitation == 0:
            rain_category = "no_rain"
        elif precipitation <= 10:
            rain_category = "rain_low"
        else:
            rain_category = "rain_high"
        
        try:
            advice_list = self.advice_data[user_type][category][rain_category].copy()
            
            # إضافة نصائح إضافية حسب درجة الحرارة
            if temperature < 5:
                advice_list.append("❄️ Extreme cold warning - take extra precautions")
            elif temperature < 10:
                advice_list.append("🥶 Cold weather - dress warmly")
            elif temperature > 40:
                advice_list.append("🔥 Extreme heat warning - stay hydrated")
            elif temperature > 30:
                advice_list.append("☀️ Hot weather - drink plenty of water")
            
            return {
                "success": True,
                "advice": advice_list,
                "rain_category": rain_category
            }
        except KeyError:
            return {
                "success": False,
                "advice": ["No specific advice available for these conditions"],
                "rain_category": rain_category
            }
    
    def get_categories(self, user_type: str) -> List[str]:
        if user_type in self.advice_data:
            return list(self.advice_data[user_type].keys())
        return []

# تهيئة الخدمات
location_service = LocationService()
weather_service = WeatherService()
advice_service = AdviceService()

# خدمة الفرونت إند
@app.get("/")
async def serve_frontend():
    return FileResponse('static/index.html')

@app.get("/api/health")
def health_check():
    return {"status": "healthy", "service": "NASA Weather Advisor"}

@app.get("/api/categories")
def get_categories(user_type: str):
    if user_type not in ["farmer", "user"]:
        raise HTTPException(status_code=400, detail="User type must be 'farmer' or 'user'")
    
    categories = advice_service.get_categories(user_type)
    return {
        "success": True,
        "user_type": user_type,
        "categories": categories
    }

@app.post("/api/analyze-weather")
def analyze_weather(request: WeatherRequest):
    # التحقق من الحقول المطلوبة
    if not all([request.country, request.city, request.date, request.user_type, request.category]):
        raise HTTPException(status_code=400, detail="All fields are required")
    
    if request.user_type not in ["farmer", "user"]:
        raise HTTPException(status_code=400, detail="User type must be 'farmer' or 'user'")
    
    # التحقق من صحة الفئة
    available_categories = advice_service.get_categories(request.user_type)
    if request.category not in available_categories:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid category for {request.user_type}. Available: {', '.join(available_categories)}"
        )
    
    # التحقق من التاريخ
    try:
        target_date = datetime.strptime(request.date, "%Y-%m-%d")
        if target_date < datetime(2020, 1, 1) or target_date > datetime(2030, 12, 31):
            raise HTTPException(status_code=400, detail="Date must be between 2020 and 2030")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    
    # الحصول على الإحداثيات
    coords = location_service.get_coordinates(request.country, request.city)
    if not coords["success"]:
        raise HTTPException(status_code=400, detail=f"Location not found: {request.city}, {request.country}")
    
    # الحصول على بيانات الطقس
    weather_data = weather_service.get_weather_data(coords["lat"], coords["lon"], request.date)
    
    # الحصول على النصائح
    advice_result = advice_service.get_advice(
        request.user_type, 
        request.category, 
        weather_data["precipitation"],
        weather_data["temperature"]
    )
    
    # الرد المناسب للفرونت إند
    return {
        "success": True,
        "location": {
            "country": request.country,
            "city": request.city,
            "coordinates": {"lat": coords["lat"], "lon": coords["lon"]},
            "display_name": coords.get("display_name", f"{request.city}, {request.country}")
        },
        "date": request.date,
        "user_type": request.user_type,
        "category": request.category,
        "weather": weather_data,
        "advice": advice_result
    }

if __name__ == "__main__":
    import uvicorn
    
    # إنشاء مجلد static إذا لم يكن موجوداً
    if not os.path.exists("static"):
        os.makedirs("static")
        print("📁 Created static directory")
    
    print("🚀 NASA Weather Advisor Server Starting...")
    print("📍 Frontend: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("⏹️  Press Ctrl+C to stop the server")
    
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")weather_cache = TTLCache(maxsize=10000, ttl=3600)

class WeatherRequest(BaseModel):
    country: str
    city: str
    date: str
    user_type: str
    category: str
    forecast_period: Optional[str] = None  # Optional: "week", "month", "year"

    @validator("country")
    def validate_country(cls, value, values):
        country_list = LocationService().get_countries()
        if value not in country_list:
            raise ValueError(f"Invalid country. Choose from: {', '.join(country_list[:10])}...")
        return value

    @validator("city")
    def validate_city(cls, value, values):
        if "country" not in values:
            raise ValueError("Country must be specified before city")
        city_list = LocationService().get_cities(values["country"])
        if value not in city_list:
            raise ValueError(f"Invalid city for {values['country']}. Choose from: {', '.join(city_list[:10])}...")
        return value

    @validator("date")
    def validate_date(cls, value):
        try:
            target_date = datetime.strptime(value, "%Y-%m-%d")
            if target_date < datetime(2020, 1, 1) or target_date > datetime(2030, 12, 31):
                raise ValueError("Date must be between 2020 and 2030")
        except ValueError as e:
            raise ValueError(f"Invalid date format: {str(e)}. Use YYYY-MM-DD")
        return value

    @validator("user_type")
    def validate_user_type(cls, value):
        if value not in ["farmer", "user"]:
            raise ValueError("User type must be 'farmer' or 'user'")
        return value

    @validator("forecast_period")
    def validate_forecast_period(cls, value):
        if value and value not in ["week", "month", "year"]:
            raise ValueError("Forecast period must be 'week', 'month', or 'year'")
        return value

class CitySuggestionRequest(BaseModel):
    country: str
    city: str

class LocationService:
    def __init__(self):
        self.country_city_map, self.city_list = self.load_geo_data()

    def load_geo_data(self) -> tuple[Dict[str, List[str]], List[str]]:
        """Load countries and cities from cities1000.txt (GeoNames)."""
        try:
            # Download from: https://download.geonames.org/export/dump/cities1000.zip
            # Columns: 1 (name), 8 (country code), 10-13 (admin codes)
            df = pd.read_csv(
                "cities1000.txt",
                sep="\t",
                header=None,
                usecols=[1, 8],
                names=["city", "country_code"],
                encoding="utf-8"
            )
            # Load country codes to names mapping
            try:
                country_df = pd.read_csv(
                    "countryInfo.txt",
                    sep="\t",
                    header=None,
                    usecols=[0, 4],
                    names=["country_code", "country_name"],
                    encoding="utf-8",
                    comment="#"  # Skip commented lines
                )
                country_map = dict(zip(country_df["country_code"], country_df["country_name"]))
            except FileNotFoundError:
                logger.warning("countryInfo.txt not found. Using country codes.")
                country_map = {code: code for code in df["country_code"].unique()}

            # Create country -> cities mapping
            country_city_map = {}
            for country_code in df["country_code"].unique():
                country_name = country_map.get(country_code, country_code)
                cities = df[df["country_code"] == country_code]["city"].unique().tolist()
                country_city_map[country_name] = sorted(cities)
            city_list = sorted(df["city"].unique().tolist())
            logger.info(f"Loaded {len(country_city_map)} countries and {len(city_list)} cities")
            return country_city_map, city_list
        except FileNotFoundError:
            logger.warning("cities1000.txt not found. Using default small list.")
            default_countries = {
                "USA": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"],
                "UK": ["London", "Manchester", "Birmingham"],
                "France": ["Paris", "Marseille"],
                "Japan": ["Tokyo"],
                "Egypt": ["Cairo"],
                "UAE": ["Dubai"]
            }
            default_cities = [city for cities in default_countries.values() for city in cities]
            return default_countries, default_cities
        except Exception as e:
            logger.error(f"Error loading GeoNames data: {str(e)}")
            return {}, []

    def get_countries(self) -> List[str]:
        """Return list of available countries."""
        return sorted(self.country_city_map.keys())

    def get_cities(self, country: str) -> List[str]:
        """Return list of cities for a given country."""
        return self.country_city_map.get(country, [])

    def get_coordinates(self, country: str, city: str) -> Dict:
        cache_key = f"{city}_{country}"
        if cache_key in location_cache:
            logger.info(f"Returning cached coordinates for {cache_key}")
            return location_cache[cache_key]

        try:
            query = f"{city}, {country}"
            url = "https://nominatim.openstreetmap.org/search"
            params = {"q": query, "format": "json", "limit": 1}
            headers = {"User-Agent": "WeatherApp/1.0"}

            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()
            if data:
                location = data[0]
                result = {
                    "success": True,
                    "lat": float(location["lat"]),
                    "lon": float(location["lon"]),
                    "display_name": location.get("display_name", f"{city}, {country}"),
                }
                location_cache[cache_key] = result
                logger.info(f"Cached coordinates for {cache_key}")
                return result
            else:
                suggestion = self.suggest_city(country, city)
                logger.warning(f"Location not found: {query}. Suggested: {suggestion}")
                return {
                    "success": False,
                    "error": f"Location not found: {city}, {country}",
                    "suggestion": suggestion
                }
        except Timeout:
            logger.error(f"Timeout error while fetching coordinates for {query}")
            return {"success": False, "error": "Geocoding request timed out"}
        except HTTPError as e:
            logger.error(f"HTTP error while fetching coordinates for {query}: {str(e)}")
            return {"success": False, "error": f"Geocoding HTTP error: {str(e)}"}
        except RequestException as e:
            logger.error(f"Request error while fetching coordinates for {query}: {str(e)}")
            return {"success": False, "error": f"Geocoding request failed: {str(e)}"}
        except Exception as e:
            logger.error(f"Unexpected error in get_coordinates for {query}: {str(e)}")
            return {"success": False, "error": f"Unexpected geocoding error: {str(e)}"}

    def suggest_city(self, country: str, city: str) -> Dict:
        """Suggest the closest matching city using fuzzy matching."""
        try:
            cities = self.get_cities(country) if country in self.country_city_map else self.city_list
            matches = process.extract(city, cities, scorer=fuzz.token_sort_ratio, limit=1)
            if matches and matches[0][1] > 50:
                suggested_city = matches[0][0]
                coords = self.get_coordinates(country, suggested_city)
                return {
                    "suggested_city": suggested_city,
                    "similarity_score": matches[0][1],
                    "coordinates": coords if coords["success"] else None
                }
            return {"suggested_city": None, "similarity_score": 0}
        except Exception as e:
            logger.error(f"Error in suggest_city for {city}, {country}: {str(e)}")
            return {"suggested_city": None, "similarity_score": 0}

class WeatherService:
    def __init__(self):
        self.nasa_url = "https://power.larc.nasa.gov/api/temporal/daily/point"

    def get_weather_data(self, lat: float, lon: float, target_date: str, forecast_period: Optional[str] = None) -> List[Dict]:
        try:
            target_dt = datetime.strptime(target_date, "%Y-%m-%d")
            current_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

            # Determine date range
            if forecast_period == "week":
                end_dt = target_dt + timedelta(days=6)
            elif forecast_period == "month":
                end_dt = target_dt + timedelta(days=29)
            elif forecast_period == "year":
                end_dt = target_dt + timedelta(days=364)
            else:
                end_dt = target_dt

            if target_dt > current_date:
                historical_start_year = target_dt.year - 1
                historical_end_year = end_dt.year - 1
                start_date = f"{historical_start_year}{target_dt.strftime('%m%d')}"
                end_date = f"{historical_end_year}{end_dt.strftime('%m%d')}"
                note = "Future period requested; using historical data from previous year"
            else:
                start_date = target_dt.strftime("%Y%m%d")
                end_date = end_dt.strftime("%Y%m%d")
                note = None

            cache_key = f"{lat}_{lon}_{start_date}_{end_date}"
            if cache_key in weather_cache:
                logger.info(f"Returning cached weather data for {cache_key}")
                return weather_cache[cache_key]

            params = {
                "parameters": "T2M,RH2M,WS2M,PRECTOT",
                "start": start_date,
                "end": end_date,
                "latitude": lat,
                "longitude": lon,
                "community": "AG",
                "format": "JSON",
            }

            response = requests.get(self.nasa_url, params=params, timeout=15)
            response.raise_for_status()

            raw_data = response.json()
            results = self._process_weather_data(raw_data, target_dt, end_dt)
            if note:
                for day in results:
                    day["note"] = note
            weather_cache[cache_key] = results
            logger.info(f"Cached weather data for {cache_key}")
            return results
        except Timeout:
            logger.error(f"Timeout error while fetching weather for {lat}, {lon}, {target_date}")
            return [self._get_default_weather(lat)]
        except HTTPError as e:
            logger.error(f"HTTP error while fetching weather for {lat}, {lon}, {target_date}: {str(e)}")
            return [self._get_default_weather(lat)]
        except RequestException as e:
            logger.error(f"Request error while fetching weather for {lat}, {lon}, {target_date}: {str(e)}")
            return [self._get_default_weather(lat)]
        except Exception as e:
            logger.error(f"Unexpected error in get_weather_data for {lat}, {lon}, {target_date}: {str(e)}")
            return [self._get_default_weather(lat)]

    def _process_weather_data(self, data: Dict, start_dt: datetime, end_dt: datetime) -> List[Dict]:
        try:
            temp_data = data["properties"]["parameter"]["T2M"]
            rh_data = data["properties"]["parameter"]["RH2M"]
            wind_data = data["properties"]["parameter"]["WS2M"]
            rain_data = data["properties"]["parameter"]["PRECTOT"]

            results = []
            current_dt = start_dt
            while current_dt <= end_dt:
                date_nasa = current_dt.strftime("%Y%m%d")
                if date_nasa in temp_data:
                    results.append({
                        "date": current_dt.strftime("%Y-%m-%d"),
                        "temperature": round(temp_data.get(date_nasa, 20), 1),
                        "humidity": round(rh_data.get(date_nasa, 50), 1),
                        "wind_speed": round(wind_data.get(date_nasa, 3), 1),
                        "precipitation": round(rain_data.get(date_nasa, 0), 1),
                        "success": True,
                    })
                else:
                    results.append(self._get_default_weather(0))
                current_dt += timedelta(days=1)
            return results
        except KeyError as e:
            logger.error(f"Key error in weather data processing: {str(e)}")
            return [self._get_default_weather(0)]
        except Exception as e:
            logger.error(f"Unexpected error in weather data processing: {str(e)}")
            return [self._get_default_weather(0)]

    def _get_default_weather(self, lat: float) -> Dict:
        base_temp = 25 if abs(lat) < 30 else 15
        return {
            "temperature": base_temp,
            "humidity": 60.0,
            "wind_speed": 5.0,
            "precipitation": 0.0,
            "success": False,
            "note": "Using estimated weather data due to API failure",
        }

class AdviceService:
    def __init__(self):
        self.advice_data = {
            "farmer": {
                "wheat": {
                    "rain_high": [
                        "Postpone irrigation today to avoid soil waterlogging",
                        "Cover the crop if possible to prevent damage",
                        "Check drainage channels to avoid water accumulation",
                    ],
                    "rain_low": [
                        "Reduce irrigation today as natural rainfall will help",
                        "Monitor soil moisture before adding extra water",
                    ],
                    "no_rain": [
                        "Irrigate normally according to your usual schedule",
                        "Check for early signs of water stress on leaves",
                    ],
                },
                "rice": {
                    "rain_high": [
                        "Ensure irrigation canals are open to prevent flooding",
                        "Check field boundaries around fields for leaks",
                        "Delay additional irrigation",
                    ],
                    "rain_low": [
                        "Light rain supports growth; reduce extra irrigation",
                        "Monitor water depth carefully in the paddy field",
                    ],
                    "no_rain": [
                        "Irrigate regularly to maintain 5-10 cm water depth",
                        "Watch for cracks in soil - early sign of dryness",
                    ],
                },
                "vegetables": {
                    "rain_high": [
                        "Cover crops with plastic sheets or greenhouse nets",
                        "Harvest ripe vegetables early to avoid spoilage",
                        "Improve drainage around rows",
                    ],
                    "rain_low": [
                        "Skip irrigation today; rainfall is sufficient",
                        "Remove excess weeds that trap moisture",
                    ],
                    "no_rain": [
                        "Irrigate in the morning or evening to reduce evaporation",
                        "Mulch soil to conserve water",
                    ],
                },
                "corn": {
                    "rain_high": [
                        "Inspect roots after heavy rain for damage",
                        "Support weak plants with soil mounding",
                    ],
                    "rain_low": [
                        "Reduce watering; rainfall is helpful",
                        "Check leaves for early fungal infections",
                    ],
                    "no_rain": [
                        "Irrigate regularly, especially during flowering stage",
                        "Apply fertilizer if leaves turn pale",
                    ],
                },
                "cotton": {
                    "rain_high": [
                        "Avoid irrigation during rainfall",
                        "Check for boll rot after heavy rain",
                    ],
                    "rain_low": [
                        "Rainfall supports growth; no extra irrigation needed",
                        "Monitor pest levels (aphids, whiteflies)",
                    ],
                    "no_rain": [
                        "Irrigate as scheduled; cotton requires consistent moisture",
                        "Inspect for leaf wilt in hot conditions",
                    ],
                },
            },
            "user": {
                "low": {
                    "rain_high": [
                        "Use plastic bags to cover shoes",
                        "Carry a small, inexpensive umbrella",
                        "Avoid drying clothes outdoors",
                    ],
                    "rain_low": [
                        "Wear a light jacket or hoodie",
                        "Carry a small plastic bag for essentials",
                    ],
                    "no_rain": [
                        "Wear regular clothes according to temperature",
                        "Stay hydrated if it's hot",
                    ],
                },
                "medium": {
                    "rain_high": [
                        "Wear a raincoat or water-resistant jacket",
                        "Carry a sturdy umbrella",
                        "Avoid slippery shoes; choose sneakers with grip",
                    ],
                    "rain_low": [
                        "A light waterproof jacket is enough",
                        "Keep a foldable umbrella in your bag",
                    ],
                    "no_rain": [
                        "Dress comfortably based on temperature",
                        "Plan outdoor activities freely",
                    ],
                },
                "high": {
                    "rain_high": [
                        "Wear waterproof jacket and waterproof shoes",
                        "Use anti-rain backpack cover",
                        "Consider driving instead of walking",
                    ],
                    "rain_low": [
                        "Stylish waterproof jacket recommended",
                        "Leather waterproof shoes are suitable",
                    ],
                    "no_rain": [
                        "Dress according to fashion and comfort",
                        "Sunglasses and light wear for sunny weather",
                    ],
                },
            },
        }

    def get_advice(self, user_type: str, category: str, precipitation: float, temperature: float) -> Dict:
        if precipitation == 0:
            rain_category = "no_rain"
        elif precipitation <= 10:
            rain_category = "rain_low"
        else:
            rain_category = "rain_high"

        try:
            advice_list = self.advice_data[user_type][category][rain_category].copy()

            if precipitation > 50:
                advice_list.append("Extreme rainfall detected; consider flood protection measures")
            elif precipitation > 30:
                advice_list.append("Heavy rain expected; prepare for potential water accumulation")

            if temperature < 5:
                advice_list.append("Extreme cold; protect crops or wear insulated clothing")
            elif temperature < 10:
                advice_list.append("Cold weather; consider frost protection for crops or warm clothing")
            elif temperature > 40:
                advice_list.append("Extreme heat; ensure hydration and avoid prolonged sun exposure")
            elif temperature > 30:
                advice_list.append("Hot weather; ensure hydration and shade")

            logger.info(f"Generated advice for {user_type}, {category}, {rain_category}, temp: {temperature}, precip: {precipitation}")
            return {
                "success": True,
                "advice": advice_list,
                "rain_category": rain_category,
            }
        except KeyError as e:
            logger.error(f"Key error in get_advice: {str(e)}")
            return {
                "success": False,
                "advice": ["No advice available for these conditions"],
                "rain_category": rain_category,
            }

    def get_categories(self, user_type: str) -> List[str]:
        return list(self.advice_data.get(user_type, {}).keys())

location_service = LocationService()
weather_service = WeatherService()
advice_service = AdviceService()

@app.get("/")
async def home():
    """Root endpoint providing API information."""
    return {
        "message": "NASA Weather Advisor API is running!",
        "status": "active",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "GET /": "This page",
            "GET /api/health": "Health check",
            "GET /docs": "API documentation",
            "GET /api/countries": "Get available countries",
            "GET /api/cities/{country}": "Get cities for a country",
            "GET /api/categories": "Get available categories",
            "POST /api/analyze-weather": "Analyze weather and get advice",
            "POST /api/suggest-city": "Suggest closest city for invalid input",
        },
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "NASA Weather Advisor",
        "timestamp": datetime.now().isoformat(),
    }

@app.get("/api/countries")
async def get_countries():
    """Get list of available countries."""
    countries = location_service.get_countries()
    return {
        "success": True,
        "countries": countries,
    }

@app.get("/api/cities/{country}")
async def get_cities(country: str):
    """Get list of cities for a given country."""
    if country not in location_service.get_countries():
        logger.warning(f"Invalid country: {country}")
        raise HTTPException(status_code=400, detail=f"Invalid country: {country}")
    cities = location_service.get_cities(country)
    return {
        "success": True,
        "country": country,
        "cities": cities,
    }

@app.get("/api/categories")
async def get_categories(user_type: str):
    """Get available categories for a user type."""
    if user_type not in ["farmer", "user"]:
        logger.warning(f"Invalid user_type: {user_type}")
        raise HTTPException(status_code=400, detail="User type must be 'farmer' or 'user'")

    categories = advice_service.get_categories(user_type)
    return {
        "success": True,
        "user_type": user_type,
        "categories": categories,
    }

@app.post("/api/suggest-city")
async def suggest_city(request: CitySuggestionRequest):
    """Suggest the closest matching city for an invalid city name."""
    suggestion = location_service.suggest_city(request.country, request.city)
    if suggestion["suggested_city"]:
        return {
            "success": True,
            "original_city": request.city,
            "suggested_city": suggestion["suggested_city"],
            "similarity_score": suggestion["similarity_score"],
            "coordinates": suggestion["coordinates"],
        }
    else:
        logger.warning(f"No city suggestion found for {request.city}, {request.country}")
        raise HTTPException(status_code=404, detail=f"No similar city found for {request.city}, {request.country}")

@app.post("/api/analyze-weather")
async def analyze_weather(request: WeatherRequest):
    """Analyze weather for a location and date, providing tailored advice."""
    # Validate required fields
    required_fields = {
        "country": request.country,
        "city": request.city,
        "date": request.date,
        "user_type": request.user_type,
        "category": request.category,
    }
    missing_fields = [field for field, value in required_fields.items() if not value.strip()]
    if missing_fields:
        logger.warning(f"Missing fields: {missing_fields}")
        raise HTTPException(status_code=400, detail=f"Missing required fields: {', '.join(missing_fields)}")

    # Get coordinates
    coords = location_service.get_coordinates(request.country, request.city)
    if not coords["success"]:
        logger.warning(f"Location not found: {request.city}, {request.country}")
        raise HTTPException(
            status_code=400,
            detail=f"Location not found: {request.city}, {request.country}. Suggestion: {coords.get('suggestion', {}).get('suggested_city', 'None')}"
        )

    # Get weather data
    weather_data = weather_service.get_weather_data(coords["lat"], coords["lon"], request.date, request.forecast_period)

    # Get advice
    if isinstance(weather_data, list):
        advice_results = [
            advice_service.get_advice(
                request.user_type,
                request.category,
                day["precipitation"],
                day["temperature"],
            ) | {"date": day["date"]}
            for day in weather_data
        ]
    else:
        advice_results = advice_service.get_advice(
            request.user_type,
            request.category,
            weather_data["precipitation"],
            weather_data["temperature"],
        )

    # Calculate statistics for multi-day forecasts
    stats = None
    if isinstance(weather_data, list) and len(weather_data) > 1:
        temps = [day["temperature"] for day in weather_data if day["success"]]
        precips = [day["precipitation"] for day in weather_data if day["success"]]
        if temps:
            stats = {
                "average_temperature": round(mean(temps), 1),
                "max_temperature": max(temps),
                "min_temperature": min(temps),
                "average_precipitation": round(mean(precips), 1),
                "max_precipitation": max(precips),
                "total_precipitation": sum(precips),
                "trend_note": "Temperatures are generally stable" if max(temps) - min(temps) < 10 else "Significant temperature variations observed"
            }

    return {
        "success": True,
        "location": {
            "country": request.country,
            "city": request.city,
            "coordinates": {"lat": coords["lat"], "lon": coords["lon"]},
            "display_name": coords.get("display_name", f"{request.city}, {request.country}"),
        },
        "date": request.date,
        "forecast_period": request.forecast_period or "single_day",
        "user_type": request.user_type,
        "category": request.category,
        "weather": weather_data,
        "advice": advice_results,
        "statistics": stats,
    }

if __name__ == "__main__":
    import uvicorn

    def open_browser():
        """Open browser after 3 seconds."""
        time.sleep(3)
        webbrowser.open("http://localhost:8000")
        webbrowser.open("http://localhost:8000/docs")

    print("=" * 60)
    print("🚀 NASA Weather Advisor Server Starting...")
    print("📍 Local URL: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("⏹️ Press Ctrl+C to stop the server")
    print("=" * 60)

    # Open browser in a separate thread
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()

    # Run the server

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
