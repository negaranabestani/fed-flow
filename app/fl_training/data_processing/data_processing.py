import os
import pandas as pd
from datetime import datetime


def process_trajectory_file(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()[6:]
        data = []
        for line in lines:
            lat, lon, _, alt, _, date_str, time_str = line.strip().split(',')
            datetime_str = f"{date_str} {time_str}"
            dt = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
            data.append((float(lat), float(lon), float(alt), dt))

    df = pd.DataFrame(data, columns=['Latitude', 'Longitude', 'Altitude', 'Datetime'])

    return df


def process_user_data(user_folder):
    trajectory_folder = os.path.join(user_folder, 'Trajectory')
    trajectory_files = [os.path.join(trajectory_folder, f) for f in os.listdir(trajectory_folder) if f.endswith('.plt')]

    user_data = pd.DataFrame()
    for file in trajectory_files:
        trajectory_data = process_trajectory_file(file)
        user_data = pd.concat([user_data, trajectory_data], ignore_index=True)

    user_data = user_data.sort_values(by='Datetime').reset_index(drop=True)

    earliest_time = user_data['Datetime'].min()
    user_data['Seconds_Since_Start'] = (user_data['Datetime'] - earliest_time).dt.total_seconds()

    return user_data


def save_user_data(ip, port, df):
    output_folder = os.path.join('processed_data', f'{ip}_{port}')  # Save directory using IP and port
    os.makedirs(output_folder, exist_ok=True)

    output_file = os.path.join(output_folder, f'{ip}_{port}_data.csv')
    df.to_csv(output_file, index=False)

    print(f"Data saved to '{output_file}'")


def process_and_save_user(data_folder, user_id, ip, port):
    user_folder = os.path.join(data_folder, user_id)
    if os.path.isdir(user_folder):
        user_data = process_user_data(user_folder)
        save_user_data(ip, port, user_data)
    else:
        print(f"User folder '{user_folder}' does not exist.")
