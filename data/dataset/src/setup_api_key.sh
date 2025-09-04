#!/usr/bin/env bash
# Usage: source concept_gen/setup_api_key.sh
# Or: . concept_gen/setup_api_key.sh

# Edit the key/model below or export beforehand.
export OPENAI_API_KEY=""
export OPENAI_MODEL="gpt-4o"

echo "OPENAI_API_KEY is set (length: ${#OPENAI_API_KEY})"
echo "OPENAI_MODEL=${OPENAI_MODEL}"


# $env:OPENAI_API_KEY=""
# $env:OPENAI_MODEL="gpt-4o"