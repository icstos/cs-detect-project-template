import config


def update_result(product_id: str, model_name: str, result: int, img_timestamp: str):
    config.r.hset(f'{product_id}_result', model_name, str(result))
    config.r.hset(f'{product_id}_result', f'{model_name}_time', img_timestamp)
