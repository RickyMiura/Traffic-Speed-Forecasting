import pandas as pd
import numpy as np
from math import cos, asin, sqrt, pi

######### Load in sensor info #########
vds_info = pd.read_csv('vds_info.csv')

######### Create dataframe for speeds and add lanes to sensor info #########
speed_data = []
lane_data = []
for ind, row in vds_info.iterrows():
    # Filepath for each week
    w1_file = str(row['vds_id']) + '_' + row['Freeway'] + row['Direction'] + '_W1' + '.csv'
    w2_file = str(row['vds_id']) + '_' + row['Freeway'] + row['Direction'] + '_W2' + '.csv'
    folder = 'SD_' + row['Freeway'] + '/'
    
    # Load in dataset for each week
    w1_df = pd.read_csv('sensor_speeds/'+folder+w1_file)
    w2_df = pd.read_csv('sensor_speeds/'+folder+w2_file)
    
    # Check that both datasets contain 1 weeks worth of 5 min intervals (1 day = 288 intervals * 7 days = 2016)
    if (len(w1_df) != 2016) or (len(w2_df) != 2016):
        print(row['vds_id'] + ' does not contain all times')
        continue
    
    # Create row representing all speeds for one sensor
    speed_row = [row['vds_id']] + list(w1_df['Speed (mph)']) + list(w2_df['Speed (mph)'])
    speed_data.append(speed_row)
    
    lane_data.append(w1_df['# Lane Points'][0]) # Assuming that # of lanes never changes
    
time_ints = list(w1_df['5 Minutes']) + list(w2_df['5 Minutes']) # Get all time intervals to use as columns
cols = ['vds_id'] + time_ints

sensor_speed = pd.DataFrame(speed_data, columns=cols).set_index('vds_id')
vds_info = vds_info.assign(Lanes=lane_data).set_index('vds_id')

# Find distance (in miles) between two coordinates 
# Source: https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula
def distance(lat1, lon1, lat2, lon2):
    r = 3956 # miles
    p = pi / 180

    a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p) * cos(lat2*p) * (1-cos((lon2-lon1)*p))/2
    return 2 * r * asin(sqrt(a))

####### Create dataframe for distances between sensors and connectivity between sensors #########
sensor_list = list(vds_info.index)
sensor_dist = pd.DataFrame(index=sensor_list, columns=sensor_list) 
sensor_conn = pd.DataFrame(index=sensor_list, columns=sensor_list)
for sen1 in sensor_list:
    for sen2 in sensor_list:
        if sen1 == sen2:
            sensor_dist.loc[sen1, sen2] = 0.0
            sensor_conn.loc[sen1, sen2] = 1
            continue
            
        # Find distances (miles) for all pairs of sensors
        sen1_lat = vds_info.loc[sen1, 'Lat']
        sen1_lon = vds_info.loc[sen1, 'Lng']
        sen2_lat = vds_info.loc[sen2, 'Lat']
        sen2_lon = vds_info.loc[sen2, 'Lng']
        
        sensor_dist.loc[sen1, sen2] = distance(sen1_lat, sen1_lon, sen2_lat, sen2_lon)
        
        # Find connectivity for all pairs of sensors
        if (vds_info.loc[sen1]['Freeway'] == vds_info.loc[sen2]['Freeway']) and (vds_info.loc[sen1]['Direction'] == vds_info.loc[sen2]['Direction']):
            sensor_conn.loc[sen1, sen2] = 1
        else:
            sensor_conn.loc[sen1, sen2] = 0

######### Create dataframe for nonconnectivity between sensors #########
non_conn = (np.ones(sensor_conn.shape) - sensor_conn).astype(int)

sensor_speed.to_csv('sensor_speed.csv', index=True)
vds_info.to_csv('vds_info_w_lanes.csv', index=True)
sensor_dist.to_csv('sensor_dist.csv', index=True)
sensor_conn.to_csv('sensor_conn.csv', index=True)
non_conn.to_csv('non_conn.csv', index=True)