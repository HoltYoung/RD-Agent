"""
Claude OAuth Proxy Server
Routes OpenAI-compatible API calls through Claude's OAuth token.
This lets RD-Agent (via LiteLLM) use your Claude Max plan subscription.

Usage:
    export CLAUDE_OAUTH_TOKEN="sk-ant-oat01-..."
    python claude_proxy.py

Then set for RD-Agent:
    OPENAI_API_BASE=http://localhost:5555/v1
    OPENAI_API_KEY=dummy
    CHAT_MODEL=claude-sonnet-4-5-20250514
"""

import json
import os
import sys
import logging
import time
from flask import Flask, request, jsonify
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("claude-proxy")

app = Flask(__name__)

OAUTH_TOKEN = os.environ.get("CLAUDE_OAUTH_TOKEN", "")
REFRESH_TOKEN = os.environ.get("CLAUDE_REFRESH_TOKEN", "")
# Claude Code CLI's official client_id
CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"

if not OAUTH_TOKEN:
    print("ERROR: Set CLAUDE_OAUTH_TOKEN environment variable")
    print("  Run 'claude setup-token' to get your token")
    sys.exit(1)

ANTHROPIC_BASE = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"

# Track token state
token_state = {
    "access_token": OAUTH_TOKEN,
    "last_refresh": time.time(),
}


def refresh_access_token():
    """Use refresh token to get a new access token."""
    if not REFRESH_TOKEN:
        logger.warning("No refresh token available, cannot refresh")
        return False
    
    try:
        resp = httpx.post(
            "https://console.anthropic.com/api/oauth/token",
            json={
                "grant_type": "refresh_token",
                "refresh_token": REFRESH_TOKEN,
                "client_id": CLIENT_ID,
            },
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            token_state["access_token"] = data["access_token"]
            token_state["last_refresh"] = time.time()
            logger.info("Successfully refreshed access token")
            return True
        else:
            logger.error(f"Refresh failed: {resp.status_code} {resp.text}")
            return False
    except Exception as e:
        logger.error(f"Refresh error: {e}")
        return False


def translate_openai_to_anthropic(data):
    """Convert OpenAI chat completion request to Anthropic messages format."""
    messages = data.get("messages", [])
    
    system = None
    filtered_messages = []
    for msg in messages:
        if msg["role"] == "system":
            if isinstance(msg["content"], str):
                system = msg["content"]
            else:
                system = msg["content"]
        else:
            filtered_messages.append({
                "role": msg["role"],
                "content": msg["content"],
            })
    
    model = data.get("model", "claude-sonnet-4-5-20250514")
    if "/" in model:
        model = model.split("/")[-1]
    
    payload = {
        "model": model,
        "max_tokens": data.get("max_tokens", 8192),
        "messages": filtered_messages,
    }
    
    if system:
        payload["system"] = system
    if data.get("temperature") is not None:
        payload["temperature"] = data["temperature"]
    if data.get("top_p") is not None:
        payload["top_p"] = data["top_p"]
    
    return payload


def translate_anthropic_to_openai(response_data, model):
    """Convert Anthropic messages response to OpenAI chat completion format."""
    content = ""
    if response_data.get("content"):
        for block in response_data["content"]:
            if block.get("type") == "text":
                content += block.get("text", "")
    
    usage = response_data.get("usage", {})
    
    return {
        "id": response_data.get("id", "chatcmpl-proxy"),
        "object": "chat.completion",
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content,
            },
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        },
    }


def make_anthropic_request(anthropic_payload, attempt=0):
    """Make request to Anthropic, with token refresh on 401."""
    resp = httpx.post(
        f"{ANTHROPIC_BASE}/v1/messages",
        headers={
            "Authorization": f"Bearer {token_state['access_token']}",
            "anthropic-version": ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        },
        json=anthropic_payload,
        timeout=300,
    )
    
    if resp.status_code == 401 and attempt == 0:
        logger.info("Got 401, attempting token refresh...")
        if refresh_access_token():
            return make_anthropic_request(anthropic_payload, attempt=1)
        else:
            logger.error("Token refresh failed")
    
    return resp


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    data = request.json
    model = data.get("model", "claude-sonnet-4-5-20250514")
    
    logger.info(f"Proxying request for model: {model}")
    
    anthropic_payload = translate_openai_to_anthropic(data)
    
    try:
        resp = make_anthropic_request(anthropic_payload)
        
        if resp.status_code != 200:
            logger.error(f"Anthropic API error {resp.status_code}: {resp.text}")
            return jsonify({"error": resp.text}), resp.status_code
        
        response_data = resp.json()
        openai_response = translate_anthropic_to_openai(response_data, model)
        
        logger.info(f"Success - {openai_response['usage']['total_tokens']} tokens")
        return jsonify(openai_response)
        
    except httpx.TimeoutException:
        logger.error("Request timed out")
        return jsonify({"error": "Request timed out"}), 504
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/v1/models", methods=["GET"])
def list_models():
    return jsonify({
        "data": [
            {"id": "claude-sonnet-4-5-20250514", "object": "model"},
            {"id": "claude-opus-4-6-20250514", "object": "model"},
            {"id": "claude-haiku-4-5-20250414", "object": "model"},
        ]
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    logger.info("Starting Claude OAuth Proxy on port 5555")
    logger.info(f"Token starts with: {token_state['access_token'][:20]}...")
    logger.info(f"Refresh token: {'set' if REFRESH_TOKEN else 'NOT SET'}")
    logger.info("Set OPENAI_API_BASE=http://localhost:5555/v1 for RD-Agent")
    app.run(host="0.0.0.0", port=5555, debug=False)
