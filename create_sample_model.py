import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Create sample training data
np.random.seed(42)
n_samples = 1000

# Generate realistic solar power data
hours = np.random.randint(0, 24, n_samples)
months = np.random.randint(1, 13, n_samples)
ambient_temp = np.random.uniform(10, 45, n_samples)
module_temp = ambient_temp + np.random.uniform(5, 15, n_samples)
irradiation = np.random.uniform(0, 1000, n_samples)

# Calculate realistic DC power based on solar physics
def calculate_solar_power(hour, irradiation, module_temp):
    # No power at night
    if hour < 6 or hour > 18:
        return 0
    
    # Peak efficiency around noon
    hour_factor = np.sin(np.pi * (hour - 6) / 12) if 6 <= hour <= 18 else 0
    
    # Temperature derating
    temp_factor = max(0, 1 - (module_temp - 25) * 0.004)
    
    # Irradiation factor
    irradiation_factor = irradiation / 1000
    
    # Base power calculation
    base_power = 2000  # 2kW system
    
    return base_power * hour_factor * temp_factor * irradiation_factor

dc_power = np.array([calculate_solar_power(h, ir, mt) for h, ir, mt in zip(hours, irradiation, module_temp)])

# Create feature matrix
X = pd.DataFrame({
    'SOURCE_KEY': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
    'AMBIENT_TEMPERATURE': ambient_temp,
    'MODULE_TEMPERATURE': module_temp,
    'IRRADIATION': irradiation,
    'hour': hours,
    'month': months
})

# Encode categorical features
X_encoded = pd.get_dummies(X, columns=['SOURCE_KEY'], prefix='SOURCE_KEY')

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_encoded, dc_power)

# Save model
joblib.dump(model, 'models/solar_model.pkl')

# Create sample dataset for analysis
sample_data = X.copy()
sample_data['DC_POWER'] = dc_power
sample_data.to_csv('data/processed/solar_final.csv', index=False)

print("✅ Sample model and dataset created successfully!")
print(f"📊 Dataset shape: {sample_data.shape}")
print(f"🎯 Model R² score: {model.score(X_encoded, dc_power):.3f}")
