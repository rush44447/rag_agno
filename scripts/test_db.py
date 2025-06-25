from psycopg import connect, OperationalError

# either read from your env or paste directly
# unused DB connections
conn_info = {
    "host": "localhost",
    "port": 5432,
    "user": "username",
    "password": "password",
    "dbname": "postgresDB",
    "connect_timeout": 5
}

try:
    conn = connect(**conn_info)
    print("✅ Connected successfully!")
    conn.close()
except OperationalError as e:
    print("❌ Connection failed:", e)