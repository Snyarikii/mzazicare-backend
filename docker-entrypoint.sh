#!/bin/sh
set -e

if [ "$SEED_DEMO_DATA" != "false" ]; then
    echo "Initializing database tables and seeding demo records..."
    python create_tables.py || true
fi

echo "Running database migrations..."
alembic upgrade head

echo "Database ready. Starting server..."
exec "$@"
