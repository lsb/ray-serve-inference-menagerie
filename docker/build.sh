#!/bin/bash

set -e

echo "Building custom Ray image with libvips support..."

docker build -f docker/Dockerfile.ray-libvips -t ray-libvips:2.8.0-py310 .

echo "Custom Ray image built successfully!"
echo ""
echo "To use this image:"
echo "1. Update RayService manifests to use 'ray-libvips:2.8.0-py310'"
echo "2. Or push to a registry: docker tag ray-libvips:2.8.0-py310 your-registry/ray-libvips:2.8.0-py310"
echo "3. Then: docker push your-registry/ray-libvips:2.8.0-py310"
