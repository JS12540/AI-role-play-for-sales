# Start Qlink Chatbot
start-bot:
    @echo "Starting Qlink Chatbot"
    docker compose -f compose.yaml --profile core up -d
    @echo "Qlink chatbot started"

# Stop entire stack
stop-all:
    @echo "Stopping entire stack"
    @docker compose --profile core down

# Stop Containers, Delete Images and Volumes
clean:
    @echo "Deleting all project images and volumes"
    @docker compose --profile core down --rmi all --volumes

# Follow logs for a specific service
follow-logs bot:
    docker logs -f ai-role-play-for-sales-bot-1; \