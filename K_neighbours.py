from sklearn import preprocessing as pr
from sklearn.neighbors import KNeighborsClassifier as knc

weather = ['Sunny','Sunny','Rainy','Overcast','Rainy','Sunny','Overcast','Rainy']
temperature = ['Hot','Mild','Cool','Mild','Cool','Hot','Mild','Cool']
play = ['No','No','Yes','Yes','Yes','No','Yes','Yes']

le_weather = pr.LabelEncoder()
le_temperature = pr.LabelEncoder()
le_label = pr.LabelEncoder()


weather_encoded = le_weather.fit_transform(weather)
# print(weather_encoded)

temperature_encoded = le_temperature.fit_transform(temperature)
# print(temperature_encoded)

label = le_label.fit_transform(play)
# print(label)

features = list(zip(weather_encoded,temperature_encoded))
# print("Encoded Feature's\n",features)

# print("Encoded Label's\n",label)

model = knc(n_neighbors=5,
            metric="euclidean")

model.fit(features,label)

input_wheather = input("Enter the current wheather(Sunny,Rainy,Overcast): ").capitalize()
input_temperature = input("Enter Temperature (Hot,Cood,Mild)").capitalize()

test_weather = le_weather.transform([input_wheather])[0]
test_temperature = le_temperature.transform([input_temperature])[0]

test_data = [[test_weather,test_temperature]]
# print(test_data)

prediction = model.predict(test_data)

print('Prediction:',le_label.inverse_transform(prediction))





