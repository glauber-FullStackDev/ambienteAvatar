#!/usr/bin/env bash
# Paste this into Vast.ai's On-start Script field when using Jupyter + SSH mode.
# Vast replaces the image ENTRYPOINT in this mode, so do not call start.sh here:
# it would start a second Jupyter server and conflict with Vast's JupyterLab.
set -u

# Match the existing LTX 2.3 Jupyter template: make Vast Docker variables
# available to terminals and run image bootstrap in the background.
env >> /etc/environment || true
mkdir -p /workspace/logs
nohup /usr/local/bin/bootstrap_a2v.sh > /workspace/logs/bootstrap.log 2>&1 &
