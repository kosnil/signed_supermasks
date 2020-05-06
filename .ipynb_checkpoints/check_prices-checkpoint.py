from bs4 import BeautifulSoup
from requests import get
import datetime
import numpy as np
import pandas as pd

#history_df = pd.read_pickle("./price_history.pkl")

url = {}
url["cpu"] = 'https://www.alternate.de/AMD/Ryzen-9-3900X-Prozessor/html/productRatings/1553398?#showRatings'
url["ram"] = "https://www.alternate.de/Corsair/DIMM-32GB-DDR4-3200-Kit-Arbeitsspeicher/html/product/1232199?"
url["cpu_cooler"] = "https://www.alternate.de/be-quiet/Dark-Rock-Pro-4-CPU-K%C3%BChler/html/product/1441526?"
url["hd"] = "https://www.alternate.de/WD/WD40EZRZ-4-TB-Festplatte/html/product/1231594?"
url["gpu"] = "https://www.alternate.de/Gainward/GeForce-RTX-2080-Ti-Phoenix-GS-Grafikkarte/html/product/1477025?"
url["mb"] = "https://www.alternate.de/ASUS/ROG-STRIX-X570-E-GAMING-Mainboard/html/product/1555257?"
url["psu"] = "https://www.alternate.de/be-quiet/Dark-Power-Pro-P11-750W-PC-Netzteil/html/product/1224608?"
url["case"] = "https://www.alternate.de/Fractal-Design/Define-7-TG-Light-Tint-Tower-Geh%C3%A4use/html/product/1602118?"
url["ssd"] = "https://www.alternate.de/Corsair/Force-MP600-1-TB-Solid-State-Drive/html/product/1557068?"

prices = {}
for key, value in url.items():
    print(f"--- Fetching Price for {key.upper()} ---")
    response = get(value)
    soup = BeautifulSoup(response.text, "html.parser")
    prices[key] = int(float(soup.find_all('span', attrs={"itemprop" : "price"})[0]["content"]))
    

total_sum = 0
for key, value in prices.items():
    print(f"{key.upper()}: {value}€")
    total_sum += value
    
print(f"Total Cost: {total_sum}€")
print(f"Total Cost of last Update: {history_df.iloc[-1].total}")
today_df = pd.DataFrame(prices, index=[datetime.datetime.now()])

new_df = today_df.append(history_df)

new_df.to_pickle("./price_history.pkl")

print("Update was added to history.")
