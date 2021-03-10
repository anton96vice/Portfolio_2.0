# Project demonstrating SQL Queries and Analysis of the Flight Data
```py
# Commented out IPython magic to ensure Python compatibility.
import pandas as pd 
import matplotlib.pyplot as plt
# %matplotlib inline
data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/SQL project/1.csv')
fuel_cost = 0.38305 #$/L https://www.indexmundi.com/commodities/?commodity=jet-fuel
'''As of January 2021, the price of Jet A1 was approximately  $450 per metric tonne. 
With a metric tonne being 1,000 KG or 2,204 lbs, this equates to about $0.45 / £0.33 per KG.
'''
'''Flight distance from Anapa to Moscow 
(Anapa Airport – Sheremetyevo International Airport) is 758 miles / 1220 kilometers / 659 nautical miles. 
Estimated flight time is 1 hour 56 minutes.'''
am_dist = 1220 #KM
am_fuel_burn = 3.46 # L/100KM / seat
am_max_seats = data.seats.max()
am_max_burn = am_max_seats * 3.46 *1220 / 100 #fuel burned at max 737

# меньше народу - больше топлива для турбин
'''
Flight distance from Anapa to Belgorod 
(Anapa Airport – Belgorod International Airport) is 391 miles / 630 kilometers / 340 nautical miles. 
Estimated flight time is 1 hour 14 minutes. '''
ab_dist = 630 #KM
ab_fuel_burn = 3.59 # L/100 km / seat
ab_max_seats = data.seats.min()
ab_max_burn = ab_max_seats * ab_fuel_burn *ab_dist / 100 #fuel burned at max for SU
print(am_max_burn, ab_max_burn)

"""# Cleaning up the data"""

print(data.info(), "---------------------\n")

print(data.head())
boolean = data.duplicated().any()
print("---------------------\n", "Duplicates exist?", boolean) #проверяем на дубликаты строк

data2 = data.drop_duplicates()
print(data2.status.nunique(), "\n----------------------------\n") #можем избавиться от статуса
data2 = data2.drop(columns= 'status') #статус у всех одинаковый
print(data2.info())

print("---------------------\n",f"Unique flights = {data2.flight_id.nunique()}" ) #убедимся что важная информация не пропала и мы всё еще работаем
#с теми же 127 полётами

# конвертируем время в datetime
timestamp_columns = ['scheduled_departure', 'scheduled_arrival','actual_departure','actual_arrival']
for time in timestamp_columns:
  data2[time] = pd.to_datetime(data2[time])
print(data2.info())

#меня интересует только длительность полёта, день и месяц
data3 = data2.copy()
for row in data2:
  data3["month"] = data2["actual_arrival"].dt.month_name()
  data3["day_of_the_week"] = data2["actual_arrival"].dt.day_name()
  data3["scheduled_duration_minutes"] = pd.to_timedelta(data2["scheduled_arrival"] - data2["scheduled_departure"]).astype('timedelta64[m]')
  data3["actual_duration_minutes"] = pd.to_timedelta(data2["actual_arrival"] - data2["actual_departure"]).astype('timedelta64[m]')
  data3["delay_in_minutes"] = pd.to_timedelta(data2["actual_departure"] - data2["scheduled_departure"]).astype('timedelta64[m]')
data3 = data3.drop(columns=timestamp_columns)

data3.head()

data3 = data3[['flight_id', 'flight_no','departure_airport','arrival_airport','arrival_city','month','day_of_the_week','scheduled_duration_minutes',
               'actual_duration_minutes', 'delay_in_minutes','book_ref','total_amount','ticket_no','fare_conditions','amount','aircraft_code','seats']]

# находим загруженность самолётов
data3['load'] = data3.groupby(by='flight_id').ticket_no.transform('count')
data3['%loaded'] = data3.load/data3.seats * 100
data3['%loaded'] = data3['%loaded'].fillna(0)

data3 = data3.drop(columns=['departure_airport','arrival_airport','scheduled_duration_minutes','ticket_no','fare_conditions','amount'])

df = data3.copy()
df = df.drop_duplicates()

"""# EDA

"""

print(df.nunique())
df.sample(20)

dff = df['arrival_city'].value_counts()
label = dff.index
size = dff.values

explode = (0, 0, 0.1)  # only "explode" the 3rd slice 

fig1, ax1 = plt.subplots()
ax1.pie(size, explode=explode, labels=label, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax1.set(title="Arrival cities")
plt.show()

print("% of flights to Novokuznetsk:", len(df[df.arrival_city == 'Novokuznetsk'].index) / len(df.index) * 100)
df_n = df[df.arrival_city == 'Novokuznetsk']

# Данные по Новокузнецку лимитированы. Запомним то, что полётов было мало, а средняя задержка была:
mean_nk_delay = df_n.delay_in_minutes.mean()
msk_delay_mean = df[df.arrival_city == "Moscow"].delay_in_minutes.mean()
b_delay_mean = df[df.arrival_city == "Belgorod"].delay_in_minutes.mean()
print("Задержка в Новокузнецк в среднем:", mean_nk_delay, "минуты")
df = df.dropna()

df.columns = ['id','no', 'city','month','day','duration','delay','ref','amount','aircraft','seats','load','%loaded']
df = df.drop(columns=['seats','ref','load'])

def pos(val):
    if val < -0.05:
      color = 'red'
    elif val > 0.05:
      color = 'green'
    else:
      color = 'black'
    return 'color: %s' % color

df.corr().style.applymap(pos)

df.groupby(['aircraft', 'month', 'day'])['amount'].mean().plot(kind = 'bar', title='Average fuel burn by month and carrier')

df

df.groupby(by='id').amount.sum()
df['amount'] = df.groupby(by=['id']).amount.transform('sum')
df.drop_duplicates(inplace=True)

df

df.loc[df['aircraft'] == '733', 'fuel_burnt'] = df['%loaded']/100 * am_max_burn
df.loc[df['aircraft'] == 'SU9', 'fuel_burnt'] = df['%loaded']/100 * ab_max_burn
df['spent_on_fuel'] = df.fuel_burnt * fuel_cost
df['profit'] = df['amount'] - df.spent_on_fuel

df

print("Можно заметить что чем меньше задержка и чем дольше полет - тем больше выгоды")
df.corr().style.applymap(pos)

print(df.profit.mean())
df.groupby(by=['no']).profit.mean().plot(kind='bar')

df.groupby(by=['aircraft','month','day']).profit.mean().plot(kind='bar')

df.groupby(by=['aircraft','month','day']).delay.sum().plot(kind='bar')
