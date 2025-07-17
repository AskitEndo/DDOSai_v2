#!/bin/bash
# Security hardening script for DDoS.AI platform

set -e

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

log "Starting security hardening process..."

# Update system packages
log "Updating system packages..."
apt-get update && apt-get upgrade -y

# Install security tools
log "Installing security tools..."
apt-get install -y fail2ban ufw auditd apparmor apparmor-utils

# Configure firewall
log "Configuring firewall..."
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
ufw allow http
ufw allow https
ufw --force enable

# Configure fail2ban
log "Configuring fail2ban..."
cat > /etc/fail2ban/jail.local << EOF
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 5

[sshd]
enabled = true
port = ssh
filter = sshd
logpath = /var/log/auth.log
maxretry = 3

[nginx-http-auth]
enabled = true
filter = nginx-http-auth
port = http,https
logpath = /var/log/nginx/error.log
EOF

systemctl restart fail2ban

# Secure SSH configuration
log "Securing SSH configuration..."
sed -i 's/#PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config
sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
sed -i 's/X11Forwarding yes/X11Forwarding no/' /etc/ssh/sshd_config
systemctl restart sshd

# Set secure file permissions
log "Setting secure file permissions..."
chmod 600 .env.production
chmod 700 scripts/*.sh
chmod -R 700 nginx/ssl
chmod -R 600 nginx/ssl/*
chmod -R 600 nginx/auth/.htpasswd

# Configure Docker security
log "Configuring Docker security..."
mkdir -p /etc/docker
cat > /etc/docker/daemon.json << EOF
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "userns-remap": "default",
  "live-restore": true,
  "userland-proxy": false,
  "no-new-privileges": true,
  "icc": false
}
EOF

systemctl restart docker

# Configure system security settings
log "Configuring system security settings..."
cat >> /etc/sysctl.conf << EOF

# Security hardening
# IP Spoofing protection
net.ipv4.conf.all.rp_filter = 1
net.ipv4.conf.default.rp_filter = 1

# Ignore ICMP broadcast requests
net.ipv4.icmp_echo_ignore_broadcasts = 1

# Disable source packet routing
net.ipv4.conf.all.accept_source_route = 0
net.ipv4.conf.default.accept_source_route = 0

# Ignore send redirects
net.ipv4.conf.all.send_redirects = 0
net.ipv4.conf.default.send_redirects = 0

# Block SYN attacks
net.ipv4.tcp_syncookies = 1
net.ipv4.tcp_max_syn_backlog = 2048
net.ipv4.tcp_synack_retries = 2
net.ipv4.tcp_syn_retries = 5

# Log Martians
net.ipv4.conf.all.log_martians = 1
net.ipv4.conf.default.log_martians = 1

# Disable IPv6 if not needed
net.ipv6.conf.all.disable_ipv6 = 1
net.ipv6.conf.default.disable_ipv6 = 1
net.ipv6.conf.lo.disable_ipv6 = 1
EOF

sysctl -p

# Set up automatic security updates
log "Setting up automatic security updates..."
apt-get install -y unattended-upgrades apt-listchanges
cat > /etc/apt/apt.conf.d/50unattended-upgrades << EOF
Unattended-Upgrade::Allowed-Origins {
    "\${distro_id}:\${distro_codename}";
    "\${distro_id}:\${distro_codename}-security";
    "\${distro_id}ESMApps:\${distro_codename}-apps-security";
    "\${distro_id}ESM:\${distro_codename}-infra-security";
};
Unattended-Upgrade::Package-Blacklist {
};
Unattended-Upgrade::DevRelease "false";
Unattended-Upgrade::Remove-Unused-Dependencies "true";
Unattended-Upgrade::Automatic-Reboot "false";
EOF

cat > /etc/apt/apt.conf.d/20auto-upgrades << EOF
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Unattended-Upgrade "1";
APT::Periodic::AutocleanInterval "7";
EOF

# Set up audit logging
log "Setting up audit logging..."
cat > /etc/audit/rules.d/audit.rules << EOF
# Delete all existing rules
-D

# Buffer Size
-b 8192

# Failure Mode
-f 1

# Monitor file system mounts
-a always,exit -S mount -S umount2 -k mount

# Monitor changes to authentication configuration files
-w /etc/group -p wa -k auth
-w /etc/passwd -p wa -k auth
-w /etc/shadow -p wa -k auth
-w /etc/sudoers -p wa -k auth

# Monitor system admin actions
-w /var/log/sudo.log -p wa -k actions

# Monitor network configuration changes
-w /etc/hosts -p wa -k network
-w /etc/network/ -p wa -k network

# Monitor Docker configuration
-w /etc/docker/ -p wa -k docker
-w /etc/docker/daemon.json -p wa -k docker

# Monitor DDoS.AI configuration
-w /opt/ddosai/.env.production -p wa -k ddosai
-w /opt/ddosai/docker-compose.prod.yml -p wa -k ddosai
-w /opt/ddosai/nginx/conf.d/ -p wa -k ddosai
EOF

systemctl restart auditd

# Set up log rotation
log "Setting up log rotation..."
cat > /etc/logrotate.d/ddosai << EOF
/opt/ddosai/backend/logs/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 root root
    sharedscripts
    postrotate
        [ -d /opt/ddosai ] && docker-compose -f /opt/ddosai/docker-compose.prod.yml exec -T backend kill -USR1 1
    endscript
}

/opt/ddosai/nginx/logs/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 root root
    sharedscripts
    postrotate
        [ -d /opt/ddosai ] && docker-compose -f /opt/ddosai/docker-compose.prod.yml exec -T nginx nginx -s reload
    endscript
}
EOF

# Set up AppArmor profiles
log "Setting up AppArmor profiles..."
aa-enforce /etc/apparmor.d/docker
aa-enforce /etc/apparmor.d/usr.sbin.nginx

# Create a security monitoring script
log "Creating security monitoring script..."
cat > /opt/security_monitor.sh << 'EOF'
#!/bin/bash

LOG_FILE="/var/log/security_monitor.log"
ALERT_EMAIL="admin@example.com"

log() {
    echo "$(date +"%Y-%m-%d %H:%M:%S") $1" >> $LOG_FILE
}

# Check for failed login attempts
failed_logins=$(grep "Failed password" /var/log/auth.log | wc -l)
if [ $failed_logins -gt 10 ]; then
    log "WARNING: High number of failed login attempts: $failed_logins"
    echo "High number of failed login attempts: $failed_logins" | mail -s "Security Alert: Failed Logins" $ALERT_EMAIL
fi

# Check for banned IPs
banned_ips=$(fail2ban-client status sshd | grep "Banned IP list" | sed 's/.*Banned IP list:\s*//')
if [ ! -z "$banned_ips" ]; then
    log "INFO: Currently banned IPs: $banned_ips"
fi

# Check disk space
disk_usage=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')
if [ $disk_usage -gt 90 ]; then
    log "WARNING: Disk usage is high: $disk_usage%"
    echo "Disk usage is high: $disk_usage%" | mail -s "Security Alert: Disk Space" $ALERT_EMAIL
fi

# Check for modified system files
if [ -f /var/lib/aide/aide.db ]; then
    aide --check >> $LOG_FILE 2>&1
    if [ $? -ne 0 ]; then
        log "WARNING: System file modifications detected"
        echo "System file modifications detected. Check the log at $LOG_FILE" | mail -s "Security Alert: File Modifications" $ALERT_EMAIL
    fi
fi

# Check Docker container security
docker_containers=$(docker ps --format "{{.Names}}")
for container in $docker_containers; do
    privileged=$(docker inspect --format='{{.HostConfig.Privileged}}' $container)
    if [ "$privileged" == "true" ]; then
        log "WARNING: Container $container is running in privileged mode"
        echo "Container $container is running in privileged mode" | mail -s "Security Alert: Privileged Container" $ALERT_EMAIL
    fi
done
EOF

chmod 700 /opt/security_monitor.sh

# Add security monitoring to crontab
log "Adding security monitoring to crontab..."
(crontab -l 2>/dev/null; echo "0 * * * * /opt/security_monitor.sh") | crontab -

# Install AIDE for file integrity monitoring
log "Installing AIDE for file integrity monitoring..."
apt-get install -y aide
aideinit
mv /var/lib/aide/aide.db.new /var/lib/aide/aide.db

# Create daily AIDE check
cat > /etc/cron.daily/aide-check << 'EOF'
#!/bin/bash
/usr/bin/aide --check
EOF
chmod +x /etc/cron.daily/aide-check

success "Security hardening completed successfully!"
log "Please review the changes and restart the system when convenient."