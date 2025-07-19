#!/bin/bash

# DDOSai IP Detection Helper Script
# This script helps you find your IP address for simulation testing

echo "🔍 DDOSai IP Detection Helper"
echo "================================="
echo ""

# Method 1: Using curl with ipify.org (same as the app)
echo "1. External IP (public internet IP):"
echo "   Using ipify.org (same service as DDOSai app):"
if command -v curl &> /dev/null; then
    EXTERNAL_IP=$(curl -s https://api.ipify.org)
    if [ $? -eq 0 ] && [ ! -z "$EXTERNAL_IP" ]; then
        echo "   ✅ $EXTERNAL_IP"
    else
        echo "   ❌ Failed to get external IP"
    fi
else
    echo "   ❌ curl not found"
fi

echo ""

# Method 2: Local network IP
echo "2. Local network IP (for local testing):"
if command -v ip &> /dev/null; then
    LOCAL_IP=$(ip route get 1.1.1.1 | grep -oP 'src \K\S+' 2>/dev/null)
    if [ ! -z "$LOCAL_IP" ]; then
        echo "   ✅ $LOCAL_IP"
    else
        echo "   ❌ Could not detect local IP"
    fi
elif command -v hostname &> /dev/null; then
    LOCAL_IP=$(hostname -I | awk '{print $1}' 2>/dev/null)
    if [ ! -z "$LOCAL_IP" ]; then
        echo "   ✅ $LOCAL_IP"
    else
        echo "   ❌ Could not detect local IP"
    fi
else
    echo "   ❌ No suitable command found"
fi

echo ""

# Method 3: All network interfaces
echo "3. All network interfaces:"
if command -v ip &> /dev/null; then
    echo "   Available IPs:"
    ip addr show | grep -E 'inet [0-9]' | grep -v '127.0.0.1' | awk '{print "   • " $2}' | sed 's/\/.*$//'
elif command -v ifconfig &> /dev/null; then
    echo "   Available IPs:"
    ifconfig | grep -E 'inet [0-9]' | grep -v '127.0.0.1' | awk '{print "   • " $2}'
else
    echo "   ❌ No suitable command found"
fi

echo ""
echo "💡 Recommendations:"
echo "   • For testing on the same machine: Use 127.0.0.1 (localhost)"
echo "   • For local network testing: Use your local IP (192.168.x.x or 10.x.x.x)"
echo "   • For internet testing: Use your external IP (⚠️  only if you own the target)"
echo ""
echo "⚠️  WARNING: Only attack systems you own or have explicit permission to test!"
echo "   Unauthorized DDoS attacks are illegal and can result in criminal charges."
echo ""
