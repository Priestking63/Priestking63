import requests


def get_exchange_rate(base_currency, target_currency):
    response = requests.get(
        f"https://api.example.com/exchange?base={base_currency}&target={target_currency}"
    )
    if response.status_code == 200:
        return response.json()["rate"]
    else:
        return None
