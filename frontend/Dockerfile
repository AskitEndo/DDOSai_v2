# Build stage
FROM node:16-alpine AS build

# Set working directory
WORKDIR /app

# Copy package files
COPY package.json package-lock.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY . .

# Build the application
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built files from build stage
COPY --from=build /app/dist /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Create a custom error page for API connectivity issues
RUN echo '<html><head><title>API Connectivity Issue</title><style>body{font-family:Arial,sans-serif;margin:40px;line-height:1.6}h1{color:#d9534f}pre{background:#f8f9fa;padding:15px;border-radius:5px}</style></head><body><h1>API Connectivity Issue</h1><p>The frontend application is running, but cannot connect to the backend API.</p><p>Please check that the backend service is running and accessible.</p></body></html>' > /usr/share/nginx/html/api-error.html

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD wget -q -O /dev/null http://localhost:80/ || exit 1

# Command to run nginx
CMD ["nginx", "-g", "daemon off;"]