FROM node:18-alpine

WORKDIR /app

# Create folders
RUN mkdir -p node_modules src public

# Copy package files
COPY frontend/package*.json ./

# Install dependencies with debug output
RUN echo "=== Installing dependencies ===" && \
    npm install && \
    echo "=== Dependencies installed, checking node_modules ===" && \
    ls -la node_modules && \
    echo "=== Install completed ==="

# Copy the rest of the frontend code
COPY frontend/ .

# Set environment for development
ENV NODE_ENV=development
ENV PORT=3000
ENV WDS_SOCKET_PORT=0

# Debug info at start
RUN echo "=== Final structure ===" && \
    ls -la && \
    echo "=== Node modules content ===" && \
    ls -la node_modules

# Start the app
CMD ["npm", "start"]