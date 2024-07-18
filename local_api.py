import json
import requests

# TODO: send a GET using the URL http://127.0.0.1:8000
url_get = "http://127.0.0.1:8000/"
r = requests.get(url_get)

# Print the status code
print(f"GET Status Code: {r.status_code}")

# Print the welcome message
print("Welcome Message:")
print(r.json()["message"])
print()

# Sample data for POST request
data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# TODO: send a POST using the data above
url_post = "http://127.0.0.1:8000/data/"
headers = {'Content-Type': 'application/json'}
r = requests.post(url_post, data=json.dumps(data), headers=headers)

# Print the status code
print(f"POST Status Code: {r.status_code}")

# Print the result
print("Prediction Result:")
print(r.json()["result"])
