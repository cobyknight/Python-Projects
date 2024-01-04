import steammarket as sm
import json

'''
item = sm.get_csgo_item('Fracture Case', currency = 'USD')
json_obj = json.dumps(item, indent = 4)
with open("sample.json", "w") as outfile:
    outfile.write(json_obj)
    
json_obj = json.loads(json_obj)

price = json_obj['lowest_price']
print(price)
'''



import requests
import json
import matplotlib.pyplot as plt

url = "https://steamcommunity.com/market/pricehistory/?country=US&currency=1&appid=730&market_hash_name=Glove%20Case%20Key"
headers = {"Cookie": "timezoneOffset=-18000,0; sessionid=79ef7225d72d40eb51cc7770; steamCountry=US%7Cfb31ec92059452bd2201bbaad6b9323b; steamLoginSecure=76561198139263088%7C%7CeyAidHlwIjogIkpXVCIsICJhbGciOiAiRWREU0EiIH0.eyAiaXNzIjogInI6MEQxOF8yMkU5RUY4RF8zMDE1NiIsICJzdWIiOiAiNzY1NjExOTgxMzkyNjMwODgiLCAiYXVkIjogWyAid2ViIiBdLCAiZXhwIjogMTY5MDQ4ODI5NCwgIm5iZiI6IDE2ODE3NjExODcsICJpYXQiOiAxNjkwNDAxMTg3LCAianRpIjogIjBEMENfMjJFOUVGOERfMTUxNDkiLCAib2F0IjogMTY5MDQwMTE4NywgInJ0X2V4cCI6IDE3MDg1OTM2NjksICJwZXIiOiAwLCAiaXBfc3ViamVjdCI6ICI5OS4xMS4yMy44MiIsICJpcF9jb25maXJtZXIiOiAiMTY2LjE5NC4xNDMuNTIiIH0.ZRfm_xM9I9vUk7QPL9KImNXKKNYP9OV_SEUlWG6mJuJddjWtm4cjqMhs9c0igC5VrE4s7CzhktJRj0U-5sJWAg; browserid=2771453880978516196; webTradeEligibility=%7B%22allowed%22%3A1%2C%22allowed_at_time%22%3A0%2C%22steamguard_required_days%22%3A15%2C%22new_device_cooldown_days%22%3A0%2C%22time_checked%22%3A1690401189%7D"}
response = requests.get(url, headers=headers)
data = response.json()

print(data)

ct = 1
yval = []
xval = []
for item in data.get('prices'):
    yval.append(item[1])
    xval.append(ct)
    ct += 1

plt.plot(xval, yval)

# naming the x axis
plt.xlabel('x - axis')
# naming the y axis
plt.ylabel('y - axis')

# giving a title to my graph
plt.title('My first graph!')

# function to show the plot
plt.show()