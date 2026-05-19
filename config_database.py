import redis

pool = redis.ConnectionPool(
    host='localhost',
    port=6379,
    db=0,
    # password='your_password',
    decode_responses=True,
    max_connections=10,  # 连接池最大连接数
    socket_keepalive=True,  # 保持 TCP 长连接
    socket_connect_timeout=5,  # 连接超时（秒）
    retry_on_timeout=True,  # 超时自动重试
)
r_cli = redis.Redis(connection_pool=pool)


mysql_info = dict(
    host="127.0.0.1",
    port=3306,
    user="user_name",
    passwd="password",
    database="database_name",
)
