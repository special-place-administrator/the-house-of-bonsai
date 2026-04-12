FROM node:20-slim AS builder

WORKDIR /app

# Copy package files
COPY package.json package-lock.json ./
COPY tsconfig.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY src/ ./src/

# Build TypeScript
RUN npm run build

FROM node:20-slim

WORKDIR /app

# Copy package files and install production deps only
COPY package.json package-lock.json ./
RUN npm ci --production

# Copy built output from builder
COPY --from=builder /app/dist ./dist

# The MCP server communicates via stdio
ENTRYPOINT ["node", "dist/server.js"]
