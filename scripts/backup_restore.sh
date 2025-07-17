#!/bin/bash
# Backup and restore script for DDoS.AI platform

set -e

# Configuration
BACKUP_DIR="./backup"
ENV_FILE=".env.production"
COMPOSE_FILE="docker-compose.prod.yml"
DATE=$(date +%Y%m%d_%H%M%S)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Log function
log() {
    echo -e "${BLUE}[$(date +"%Y-%m-%d %H:%M:%S")]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    error "Please run as root or with sudo"
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    error "Docker is not installed. Please install Docker first."
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    error "Docker Compose is not installed. Please install Docker Compose first."
fi

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

# Load environment variables
if [ -f "$ENV_FILE" ]; then
    log "Loading environment variables from $ENV_FILE"
    set -a
    source "$ENV_FILE"
    set +a
else
    warning "Environment file $ENV_FILE not found. Using default environment variables."
fi

# Function to create a backup
create_backup() {
    log "Creating backup..."
    
    # Create backup directory for this backup
    BACKUP_PATH="$BACKUP_DIR/backup_$DATE"
    mkdir -p "$BACKUP_PATH"
    
    # Backup PostgreSQL database
    log "Backing up PostgreSQL database..."
    docker-compose -f "$COMPOSE_FILE" exec -T postgres pg_dumpall -c -U "$POSTGRES_USER" | gzip > "$BACKUP_PATH/database.sql.gz"
    
    # Backup Redis data
    log "Backing up Redis data..."
    docker-compose -f "$COMPOSE_FILE" exec -T redis redis-cli -a "$REDIS_PASSWORD" --rdb "$BACKUP_PATH/redis.rdb"
    
    # Backup application data
    log "Backing up application data..."
    docker run --rm --volumes-from $(docker-compose -f "$COMPOSE_FILE" ps -q backend) -v "$BACKUP_PATH:/backup" alpine tar -czf /backup/backend_data.tar.gz -C /app/data .
    docker run --rm --volumes-from $(docker-compose -f "$COMPOSE_FILE" ps -q backend) -v "$BACKUP_PATH:/backup" alpine tar -czf /backup/backend_models.tar.gz -C /app/models .
    
    # Backup environment file
    if [ -f "$ENV_FILE" ]; then
        log "Backing up environment file..."
        cp "$ENV_FILE" "$BACKUP_PATH/.env.backup"
    fi
    
    # Create a single archive of all backups
    log "Creating final backup archive..."
    tar -czf "$BACKUP_DIR/ddosai_backup_$DATE.tar.gz" -C "$BACKUP_DIR" "backup_$DATE"
    
    # Remove temporary backup directory
    rm -rf "$BACKUP_PATH"
    
    success "Backup completed: $BACKUP_DIR/ddosai_backup_$DATE.tar.gz"
}

# Function to restore from a backup
restore_backup() {
    if [ -z "$1" ]; then
        error "Please specify a backup file to restore"
    fi
    
    BACKUP_FILE="$1"
    
    if [ ! -f "$BACKUP_FILE" ]; then
        error "Backup file $BACKUP_FILE not found"
    fi
    
    log "Restoring from backup: $BACKUP_FILE"
    
    # Create temporary directory for extraction
    TEMP_DIR="$BACKUP_DIR/temp_restore_$DATE"
    mkdir -p "$TEMP_DIR"
    
    # Extract backup archive
    log "Extracting backup archive..."
    tar -xzf "$BACKUP_FILE" -C "$TEMP_DIR"
    
    # Find the extracted backup directory
    BACKUP_SUBDIR=$(find "$TEMP_DIR" -type d -name "backup_*" | head -n 1)
    
    if [ -z "$BACKUP_SUBDIR" ]; then
        error "Invalid backup format: backup directory not found"
    fi
    
    # Stop services
    log "Stopping services..."
    docker-compose -f "$COMPOSE_FILE" down
    
    # Restore environment file if it exists
    if [ -f "$BACKUP_SUBDIR/.env.backup" ]; then
        log "Restoring environment file..."
        cp "$BACKUP_SUBDIR/.env.backup" "$ENV_FILE"
        
        # Reload environment variables
        set -a
        source "$ENV_FILE"
        set +a
    fi
    
    # Start database and Redis services
    log "Starting database and Redis services..."
    docker-compose -f "$COMPOSE_FILE" up -d postgres redis
    
    # Wait for services to start
    log "Waiting for services to start..."
    sleep 10
    
    # Restore PostgreSQL database
    log "Restoring PostgreSQL database..."
    gunzip -c "$BACKUP_SUBDIR/database.sql.gz" | docker-compose -f "$COMPOSE_FILE" exec -T postgres psql -U "$POSTGRES_USER"
    
    # Restore Redis data
    log "Restoring Redis data..."
    docker-compose -f "$COMPOSE_FILE" stop redis
    docker run --rm --volumes-from $(docker-compose -f "$COMPOSE_FILE" ps -q redis) -v "$BACKUP_SUBDIR:/backup" alpine sh -c "rm -rf /data/* && cp /backup/redis.rdb /data/"
    docker-compose -f "$COMPOSE_FILE" start redis
    
    # Restore application data
    log "Restoring application data..."
    docker run --rm --volumes-from $(docker-compose -f "$COMPOSE_FILE" ps -q backend) -v "$BACKUP_SUBDIR:/backup" alpine sh -c "rm -rf /app/data/* && tar -xzf /backup/backend_data.tar.gz -C /app/data"
    docker run --rm --volumes-from $(docker-compose -f "$COMPOSE_FILE" ps -q backend) -v "$BACKUP_SUBDIR:/backup" alpine sh -c "rm -rf /app/models/* && tar -xzf /backup/backend_models.tar.gz -C /app/models"
    
    # Start all services
    log "Starting all services..."
    docker-compose -f "$COMPOSE_FILE" up -d
    
    # Clean up
    log "Cleaning up temporary files..."
    rm -rf "$TEMP_DIR"
    
    success "Restore completed successfully"
}

# Function to list available backups
list_backups() {
    log "Available backups:"
    
    if [ ! "$(ls -A "$BACKUP_DIR")" ]; then
        echo "No backups found in $BACKUP_DIR"
        return
    fi
    
    echo "----------------------------------------"
    echo "Filename                      | Size    | Date"
    echo "----------------------------------------"
    
    for backup in "$BACKUP_DIR"/ddosai_backup_*.tar.gz; do
        if [ -f "$backup" ]; then
            filename=$(basename "$backup")
            size=$(du -h "$backup" | cut -f1)
            date_str=$(echo "$filename" | grep -o '[0-9]\{8\}_[0-9]\{6\}' | sed 's/\([0-9]\{4\}\)\([0-9]\{2\}\)\([0-9]\{2\}\)_\([0-9]\{2\}\)\([0-9]\{2\}\)\([0-9]\{2\}\)/\1-\2-\3 \4:\5:\6/')
            printf "%-30s | %-7s | %s\n" "$filename" "$size" "$date_str"
        fi
    done
    
    echo "----------------------------------------"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  backup              Create a new backup"
    echo "  restore [filename]  Restore from a backup"
    echo "  list                List available backups"
    echo "  help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 backup                         # Create a new backup"
    echo "  $0 restore backup/ddosai_backup_20230717_101530.tar.gz  # Restore from a specific backup"
    echo "  $0 list                           # List available backups"
}

# Main script logic
case "$1" in
    backup)
        create_backup
        ;;
    restore)
        restore_backup "$2"
        ;;
    list)
        list_backups
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        show_usage
        exit 1
        ;;
esac