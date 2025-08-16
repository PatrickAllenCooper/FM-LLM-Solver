#!/bin/sh

# Health check script for frontend service
curl -f http://localhost:8080/health || exit 1

