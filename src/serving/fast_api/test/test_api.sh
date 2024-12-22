#!/bin/bash

URL="http://127.0.0.1:8000/predict/"
FILE_PATH="request_input.json"

curl -X POST \
  "$URL" \
  -H "Content-Type: application/json" \
  -d @"$FILE_PATH"
