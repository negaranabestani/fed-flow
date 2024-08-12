import os
import threading
import time

import pandas as pd

from app.entity.node import Node


def load_user_data(user_id, data_folder='/fed-flow/app/util/processed_data'):
    user_folder = os.path.join(data_folder, user_id)
    csv_file = os.path.join(user_folder, f'{user_id}_data.csv')
    user_data = pd.read_csv(csv_file, parse_dates=['Datetime'])

    return user_data


def simulate_real_time_update(node: Node, user_data):
    max_seconds = user_data['Seconds_Since_Start'].max()

    for current_second in range(0, int(max_seconds) + 1):
        current_data = user_data[user_data['Seconds_Since_Start'] <= current_second].iloc[-1]

        node.update_coordinates(new_latitude=current_data['Latitude'],
                                new_longitude=current_data['Longitude'],
                                new_altitude=current_data['Altitude'],
                                new_seconds_since_start=current_second)

        time.sleep(1)


def start_simulation_thread(node: Node, user_data: pd.DataFrame):
    simulation_thread = threading.Thread(target=simulate_real_time_update, args=(node, user_data))
    simulation_thread.start()
