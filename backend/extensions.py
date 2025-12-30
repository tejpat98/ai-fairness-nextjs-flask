import os
import redis
from rq import Queue

# Connect to Redis
# We use 'redis' as the default host to match the docker-compose service name
redis_host = os.environ.get('REDIS_HOST', 'redis')
redis_conn = redis.Redis(host=redis_host, port=6379, db=0)

# Task queue instance
task_queue = Queue(connection=redis_conn)
