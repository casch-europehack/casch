import random
from datetime import datetime, timedelta
from typing import Dict, Any
from services.storage import load_from_db

class MockCO2Calculator:
    def calculate(self, data: Dict[str, Any], location: str) -> Dict[str, Any]:
        # Mock calculation: return some dummy CO2 emissions per hour
        # We will add the actual calculation later
        emissions_per_hour = []
        times = []
        
        current_time = datetime.now()
        count = 0
        total_emissions = 0.0
        
        # Generate a lot of data using a while loop
        while count < 200:
            times.append(current_time.isoformat())
            emission = round(random.uniform(8.0, 15.0), 2)
            emissions_per_hour.append(emission)
            total_emissions += emission
            
            current_time += timedelta(hours=1)
            count += 1

        return {
            "emissions_per_hour": emissions_per_hour,
            "times": times,
            "total_emissions": round(total_emissions, 2),
            "location": location
        }

class CO2ProxyService:
    def __init__(self):
        self.calculator = MockCO2Calculator()

    def get_co2_emissions(self, file_hash: str, location: str) -> Dict[str, Any]:
        data = load_from_db(file_hash)
        if not data:
            raise ValueError("Data not found for the given file hash")
        
        return self.calculator.calculate(data, location)

co2_service = CO2ProxyService()
