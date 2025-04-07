import requests
import numpy as np

# Génère des données de test aléatoires (adaptez à votre cas)
test_data = np.random.rand(1, 10, 5).tolist()  # 5 features

response = requests.post(
    "http://localhost:5000/predict",
    json={"input": test_data},
    headers={"Content-Type": "application/json"}
)

print(response.status_code)
print(response.json())