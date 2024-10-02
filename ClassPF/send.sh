#!/bin/bash

# Local directory containing the files you want to send
LOCAL_DIR="/Users/mahdipashaei/Desktop/NSPF"

# Remote cluster details
REMOTE_USER="pasha77"              # Your username on the cluster
REMOTE_HOST="beluga.computecanada.ca"   # The hostname of the cluster
# REMOTE_HOST="cedar.computecanada.ca"   # The hostname of the cluster
REMOTE_DIR="/home/pasha77/scratch"      # Remote directory where the files should be sent

# Sync only .py files and job.sh from the local directory to the remote directory
rsync -avz --progress --include='*.py' --include='job.sh' --exclude='*' "$LOCAL_DIR/" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR"

# Explanation of options:
# -a: archive mode (preserves permissions, timestamps, symbolic links, etc.)
# -v: verbose mode (provides detailed output)
# -z: compress the file data during the transfer
# --progress: show progress during transfer
# --include: specify which files to include in the transfer
# --exclude: exclude all other files that don't match the includes
