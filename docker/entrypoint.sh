#!/bin/sh
set -e
mkdir -p "$AUTOPILOT_VAR_DIR"
# Train the classifier on first boot so the API never starts on the heuristic.
if [ ! -f "$AUTOPILOT_CLASSIFIER" ] && [ "$1" != "dashboard" ]; then
  autopilot train > /dev/null && echo "trained classifier -> $AUTOPILOT_CLASSIFIER"
fi
case "$1" in
  serve)     exec autopilot serve --host 0.0.0.0 --port 8000 ;;
  worker)    exec autopilot worker ;;
  dashboard) exec streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true ;;
  *)         exec autopilot "$@" ;;
esac
