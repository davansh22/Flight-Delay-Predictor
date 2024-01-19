
from scipy.stats import lognorm
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import random
def uii( in1,in2,in3):
    df = pd.read_csv('flights.csv')
    airlinedf = pd.read_csv('airlines.csv')
    airportdf = pd.read_csv('airports.csv')
    sample_size = 100000
    data_subset = df.head(sample_size)


    airlinedict = dict(zip(airlinedf['AIRLINE'],airlinedf['IATA_CODE']))
    airportdict = dict(zip(airportdf['CITY'],airportdf['IATA_CODE']))
    var1 = airlinedict[in1]
    var3 = airportdict[in2]
    var4 = airportdict[in3]

    columns_to_encode = ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']

    label_encoder = LabelEncoder()
    encoding_mapping = {}

    for column in columns_to_encode:
        data_subset[column] = label_encoder.fit_transform(data_subset[column])
        encoding_mapping[column] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

    var2 = encoding_mapping['AIRLINE'][var1]
    var5 = encoding_mapping['ORIGIN_AIRPORT'][var3]
    var6 = encoding_mapping['DESTINATION_AIRPORT'][var4]
    X_train_features = data_subset[['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']]
    y_departure_delay = data_subset['DEPARTURE_DELAY']

    
    y_departure_delay = y_departure_delay.fillna(0)  
    from sklearn.model_selection import train_test_split

# Assuming 'data_subset' is your dataset
    X_train, X_test, y_train, y_test = train_test_split(
        data_subset[['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']],
        y_departure_delay,
        test_size=0.2, 
        random_state=random.randint(0,100)  
    )
    
    tree_model = DecisionTreeRegressor()

        
    tree_model.fit(X_train, y_train)

   
    input_data = pd.DataFrame({'AIRLINE': [var2], 'ORIGIN_AIRPORT': [var5], 'DESTINATION_AIRPORT': [var6]})  

    departure_delay_predictions = tree_model.predict(input_data)
    print("Predicted Arrival Delays:", departure_delay_predictions)
    return departure_delay_predictions[0]