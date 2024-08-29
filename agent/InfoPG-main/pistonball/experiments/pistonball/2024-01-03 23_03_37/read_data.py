import pickle

with open("data.pkl", "rb") as f:
    data = pickle.load(f)
num = 0
for i in data:
    if num % 20 == 0:
        print(i["红无人机1"][0], i["红无人机3"][0], i["红无人机1"][0] + i["红无人机3"][0], i["红无人机1"][1])
    num += 1