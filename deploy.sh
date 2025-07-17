#!/bin/bash
# Production deployment script for DDoS.AI platform

set -e

# Configuration
ENV_FILE=".env.production"
COMPOSE_FILE="docker-compose.prod.yml"
BACKUP_DIR="./backup"
SSL_DIR="./nginx/ssl"
LOG_FILE="deploy_$(date +%Y%m%d_%H%M%S).log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Log function
log() {
    echo -e "${BLUE}[$(date +"%Y-%m-%d %H:%M:%S")]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
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

# Check if environment file exists
if [ ! -f "$ENV_FILE" ]; then
    warning "Environment file $ENV_FILE not found. Creating from template..."
    cat > "$ENV_FILE" << EOF
# DDoS.AI Production Environment Variables
POSTGRES_USER=ddosai
POSTGRES_PASSWORD=$(openssl rand -base64 16)
REDIS_PASSWORD=$(openssl rand -base64 16)
API_KEY=$(openssl rand -base64 32)
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=$(openssl rand -base64 12)
EOF
    success "Created $ENV_FILE with random passwords. Please review and update if needed."
fi

# Load environment variables
log "Loading environment variables from $ENV_FILE"
set -a
source "$ENV_FILE"
set +a

# Create directories if they don't exist
log "Creating required directories"
mkdir -p "$BACKUP_DIR"
mkdir -p "$SSL_DIR"
mkdir -p "./nginx/auth"

# Check if SSL certificates exist
if [ ! -d "$SSL_DIR/ddosai.example.com" ] || [ ! -f "$SSL_DIR/ddosai.example.com/fullchain.pem" ]; then
    warning "SSL certificates not found. For production, you should use real SSL certificates."
    warning "You can use Let's Encrypt to generate free SSL certificates."
    warning "For now, generating self-signed certificates for testing..."
    
    # Create directories for certificates
    mkdir -p "$SSL_DIR/ddosai.example.com"
    mkdir -p "$SSL_DIR/api.ddosai.example.com"
    mkdir -p "$SSL_DIR/grafana.ddosai.example.com"
    mkdir -p "$SSL_DIR/prometheus.ddosai.example.com"
    
    # Generate DH parameters
    openssl dhparam -out "$SSL_DIR/dhparam.pem" 2048
    
    # Generate self-signed certificates for each domain
    for domain in ddosai.example.com api.ddosai.example.com grafana.ddosai.example.com prometheus.ddosai.example.com; do
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout "$SSL_DIR/$domain/privkey.pem" \
            -out "$SSL_DIR/$domain/fullchain.pem" \
            -subj "/CN=$domain" \
            -addext "subjectAltName=DNS:$domain"
    done
    
    success "Generated self-signed certificates"
fi

# Create htpasswd file for Prometheus
if [ ! -f "./nginx/auth/.htpasswd" ]; then
    log "Creating htpasswd file for Prometheus"
    docker run --rm httpd:alpine htpasswd -Bbn admin "$(openssl rand -base64 12)" > "./nginx/auth/.htpasswd"
fi

# Pull latest images
log "Pulling latest Docker images"
docker-compose -f "$COMPOSE_FILE" pull

# Backup existing data
if [ -d "/var/lib/docker/volumes/ddosai_postgres_data" ]; then
    log "Backing up existing data"
    BACKUP_FILE="$BACKUP_DIR/backup_$(date +%Y%m%d_%H%M%S).tar.gz"
    docker-compose -f "$COMPOSE_FILE" exec -T postgres pg_dumpall -c -U "$POSTGRES_USER" | gzip > "$BACKUP_FILE"
    success "Database backup created: $BACKUP_FILE"
fi

# Stop and remove existing containers
log "Stopping existing containers"
docker-compose -f "$COMPOSE_FILE" down

# Start new containers
log "Starting containers"
docker-compose -f "$COMPOSE_FILE" up -d

# Wait for services to start
log "Waiting for services to start..."
sleep 10

# Check if services are running
log "Checking service health"
docker-compose -f "$COMPOSE_FILE" ps

# Run database migrations if needed
log "Running database migrations"
docker-compose -f "$COMPOSE_FILE" exec -T backend python -m alembic upgrade head

# Final checks
log "Performing final checks"
for service in backend frontend redis postgres prometheus grafana nginx; do
    if ! docker-compose -f "$COMPOSE_FILE" exec -T $service echo "Service $service is running"; then
        warning "Service $service may not be running properly"
    fi
done

# Setup firewall rules
log "Setting up firewall rules"
if command -v ufw &> /dev/null; then
    ufw allow 80/tcp
    ufw allow 443/tcp
    success "Firewall rules updated"
else
    warning "UFW not found. Please configure your firewall manually to allow ports 80 and 443"
fi

# Print success message
success "Deployment completed successfully!"
echo -e "${GREEN}=============================================${NC}"
echo -e "${GREEN}DDoS.AI Platform is now running in production mode${NC}"
echo -e "${GREEN}=============================================${NC}"
echo -e "Access the platform at: ${BLUE}https://ddosai.example.com${NC}"
echo -e "API is available at: ${BLUE}https://api.ddosai.example.com${NC}"
echo -e "Grafana dashboards: ${BLUE}https://grafana.ddosai.example.com${NC}"
echo -e "${YELLOW}NOTE: Update your DNS records to point to this server${NC}"
echo -e "${YELLOW}NOTE: For production use, replace self-signed certificates with real ones${NC}"
echo -e "${GREEN}=============================================${NC}"

# Print log file location
echo -e "Deployment log saved to: ${BLUE}$LOG_FILE${NC}"