from locust import HttpUser, task, between


class MuseumAPIUser(HttpUser):
    wait_time = between(0.1, 0.5)
    payload = {
        "age_years": 150,
        "size_cm": 30,
        "estimated_value_usd": 500,
        "organic": 0,
        "has_inscription": 1,
        "excavation_find": 1,
        "natural_specimen": 0,
        "artwork": 0,
    }

    @task(4)
    def predict_v1(self):
        self.client.post("/predict", json=self.payload, name="/predict")

    @task(2)
    def predict_v2(self):
        self.client.post("/v2/predict", json=self.payload, name="/v2/predict")

    @task(1)
    def health_check(self):
        self.client.get("/health", name="/health")