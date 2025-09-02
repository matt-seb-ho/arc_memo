#!/usr/bin/env bash
# Usage: source concept_gen/setup_api_key.sh
# Or: . concept_gen/setup_api_key.sh

# Edit the key/model below or export beforehand.
export OPENAI_API_KEY="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"
export OPENAI_MODEL="gpt-4o"

echo "OPENAI_API_KEY is set (length: ${#OPENAI_API_KEY})"
echo "OPENAI_MODEL=${OPENAI_MODEL}"


# $env:OPENAI_API_KEY="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"
# $env:OPENAI_MODEL="gpt-5"