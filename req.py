import requests

X = [6.4, 2.9, 5.2, 1.6]
url_string = f'http://localhost:8000/predict?f0={X[0]}&f1={X[1]}&f2={X[2]}&f3={X[3]}'
r = requests.get(url_string)
print(r.json())