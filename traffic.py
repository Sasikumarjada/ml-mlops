import json
import random
import time
import urllib.error
import urllib.request

URL = "http://localhost:8000"

for i in range(1, 151):
    if i % 10 == 0:
        payload = {"sepal_length": -1}  # invalid on purpose, creates a 422 error
    else:
        payload = {
            "sepal_length": round(random.uniform(4.3, 7.9), 1),
            "sepal_width": round(random.uniform(2.0, 4.4), 1),
            "petal_length": round(random.uniform(1.0, 6.9), 1),
            "petal_width": round(random.uniform(0.1, 2.5), 1),
        }
    req = urllib.request.Request(
        URL + "/predict",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(req).read()
    except urllib.error.HTTPError as e:
        print(f"request {i}: HTTP {e.code}")
    time.sleep(0.2)

print("done")