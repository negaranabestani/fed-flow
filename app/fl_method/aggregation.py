from app.config.logger import fed_logger


def fed_avg(zero_model, w_local_list, total_data_size):
    # Log the initial state of the inputs
    fed_logger.info(f"Starting federated averaging. Total data size: {total_data_size}")
    fed_logger.info(f"Local weights list: {w_local_list}")

    # Check if w_local_list is not empty
    if not w_local_list:
        fed_logger.error("w_local_list is empty. Cannot proceed with aggregation.")
        return zero_model

    # Ensure the first element is a tuple with the expected structure
    if len(w_local_list) > 0 and isinstance(w_local_list[0], tuple) and isinstance(w_local_list[0][0], dict):
        keys = w_local_list[0][0].keys()
    else:
        fed_logger.error("Unexpected structure of w_local_list. Expected a list of tuples with dicts.")
        return zero_model

    # Log the keys being processed
    fed_logger.info(f"Keys in the model to aggregate: {keys}")

    for k in keys:
        for w in w_local_list:
            beta = float(w[1]) / float(total_data_size)
            if 'num_batches_tracked' in k:
                zero_model[k] = w[0][k]
            else:
                zero_model[k] += (w[0][k] * beta)

    fed_logger.info("Aggregation completed successfully.")
    return zero_model

