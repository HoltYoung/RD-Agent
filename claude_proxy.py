"""
Claude OAuth Proxy Server
Routes OpenAI-compatible API calls through Claude's OAuth token,
mimicking Claude Code's exact auth headers so api.anthropic.com accepts them.

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
import uuid
from flask import Flask, request, jsonify, Response
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("claude-proxy")

app = Flask(__name__)

OAUTH_TOKEN = os.environ.get("CLAUDE_OAUTH_TOKEN", "")
REFRESH_TOKEN = os.environ.get("CLAUDE_REFRESH_TOKEN", "")
CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"

if not OAUTH_TOKEN:
    print("ERROR: Set CLAUDE_OAUTH_TOKEN environment variable")
    print("  Find it in ~/.claude/credentials.json or run: cat ~/.claude/credentials.json | python -c \"import sys,json; d=json.load(sys.stdin); print(list(d.values())[0]['oauth']['accessToken'])\"")
    sys.exit(1)

ANTHROPIC_BASE = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"
CLAUDE_CODE_VERSION = "2.1.2"

# The exact beta flags OpenClaw/Claude Code sends
BETA_FLAGS = "claude-code-20250219,oauth-2025-04-20,fine-grained-tool-streaming-2025-05-14,interleaved-thinking-2025-05-14"

# Required system prompt prefix for OAuth tokens
CLAUDE_CODE_SYSTEM_PREFIX = "You are Claude Code, Anthropic's official CLI for Claude."

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
            "https://console.anthropic.com/v1/oauth/token",
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
            logger.info("Refreshed access token successfully")
            return True
        else:
            logger.error(f"Refresh failed: {resp.status_code} {resp.text}")
            return False
    except Exception as e:
        logger.error(f"Refresh error: {e}")
        return False


def get_anthropic_headers():
    """Build the exact headers Claude Code sends for OAuth auth."""
    return {
        "Authorization": f"Bearer {token_state['access_token']}",
        "anthropic-version": ANTHROPIC_VERSION,
        "anthropic-beta": BETA_FLAGS,
        "user-agent": f"claude-cli/{CLAUDE_CODE_VERSION} (external, cli)",
        "x-app": "cli",
        "accept": "application/json",
        "anthropic-dangerous-direct-browser-access": "true",
        "Content-Type": "application/json",
    }


def translate_openai_to_anthropic(data):
    """Convert OpenAI chat completion request to Anthropic messages format."""
    messages = data.get("messages", [])

    system_parts = []
    filtered_messages = []
    for msg in messages:
        if msg["role"] == "system":
            text = msg["content"] if isinstance(msg["content"], str) else json.dumps(msg["content"])
            system_parts.append(text)
        else:
            filtered_messages.append({
                "role": msg["role"],
                "content": msg["content"],
            })

    model = data.get("model", "claude-sonnet-4-5-20250514")
    if "/" in model:
        model = model.split("/")[-1]

    # Build system prompt: MUST start with Claude Code identity for OAuth
    user_system = "\n\n".join(system_parts) if system_parts else ""
    system_blocks = [
        {"type": "text", "text": CLAUDE_CODE_SYSTEM_PREFIX},
    ]
    if user_system:
        system_blocks.append({"type": "text", "text": user_system})

    payload = {
        "model": model,
        "max_tokens": data.get("max_tokens", 8192),
        "messages": filtered_messages,
        "system": system_blocks,
    }

    if data.get("temperature") is not None:
        payload["temperature"] = data["temperature"]
    if data.get("top_p") is not None:
        payload["top_p"] = data["top_p"]
    if data.get("stream"):
        payload["stream"] = True

    return payload, model


def anthropic_to_openai_response(response_data, model):
    """Convert Anthropic messages response to OpenAI chat completion format."""
    content = ""
    if response_data.get("content"):
        for block in response_data["content"]:
            if block.get("type") == "text":
                content += block.get("text", "")

    usage = response_data.get("usage", {})
    stop = response_data.get("stop_reason", "end_turn")
    finish = "stop" if stop == "end_turn" else "length" if stop == "max_tokens" else "stop"

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": finish,
        }],
        "usage": {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        },
    }


def make_request(anthropic_payload, stream=False, attempt=0):
    """Make request to Anthropic with Claude Code headers. Retry once on 401."""
    headers = get_anthropic_headers()
    resp = httpx.post(
        f"{ANTHROPIC_BASE}/v1/messages",
        headers=headers,
        json=anthropic_payload,
        timeout=300,
    )
    if resp.status_code == 401 and attempt == 0:
        logger.info("401 received, refreshing token...")
        if refresh_access_token():
            return make_request(anthropic_payload, stream, attempt=1)
    return resp


def make_streaming_request(anthropic_payload, attempt=0):
    """Make streaming request to Anthropic."""
    headers = get_anthropic_headers()
    try:
        with httpx.stream(
            "POST",
            f"{ANTHROPIC_BASE}/v1/messages",
            headers=headers,
            json=anthropic_payload,
            timeout=300,
        ) as resp:
            if resp.status_code == 401 and attempt == 0:
                logger.info("401 on stream, refreshing token...")
                if refresh_access_token():
                    yield from make_streaming_request(anthropic_payload, attempt=1)
                    return
            if resp.status_code != 200:
                error_body = resp.read().decode()
                logger.error(f"Stream error {resp.status_code}: {error_body}")
                yield f"data: {json.dumps({'error': error_body})}\n\n"
                return

            # Parse Anthropic SSE and re-emit as OpenAI SSE
            content_so_far = ""
            comp_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                etype = event.get("type", "")
                if etype == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text = delta.get("text", "")
                        content_so_far += text
                        chunk = {
                            "id": comp_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "choices": [{
                                "index": 0,
                                "delta": {"content": text},
                                "finish_reason": None,
                            }],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                elif etype == "message_delta":
                    stop = event.get("delta", {}).get("stop_reason", "end_turn")
                    finish = "stop" if stop == "end_turn" else "length"
                    chunk = {
                        "id": comp_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": finish,
                        }],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

            yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    data = request.json
    is_stream = data.get("stream", False)

    anthropic_payload, model = translate_openai_to_anthropic(data)
    logger.info(f"Proxying: model={model}, stream={is_stream}, msgs={len(anthropic_payload['messages'])}")

    if is_stream:
        anthropic_payload["stream"] = True
        return Response(
            make_streaming_request(anthropic_payload),
            content_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        resp = make_request(anthropic_payload)
        if resp.status_code != 200:
            logger.error(f"Anthropic error {resp.status_code}: {resp.text}")
            return jsonify({"error": {"message": resp.text, "type": "api_error"}}), resp.status_code

        response_data = resp.json()
        openai_resp = anthropic_to_openai_response(response_data, model)
        logger.info(f"OK - {openai_resp['usage']['total_tokens']} tokens")
        return jsonify(openai_resp)

    except httpx.TimeoutException:
        return jsonify({"error": {"message": "Timeout", "type": "timeout"}}), 504
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({"error": {"message": str(e), "type": "server_error"}}), 500


@app.route("/v1/models", methods=["GET"])
def list_models():
    return jsonify({
        "data": [
            {"id": "claude-sonnet-4-5-20250514", "object": "model", "owned_by": "anthropic"},
            {"id": "claude-opus-4-6-20250514", "object": "model", "owned_by": "anthropic"},
            {"id": "claude-haiku-4-5-20250414", "object": "model", "owned_by": "anthropic"},
        ]
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "token_prefix": token_state["access_token"][:20]})


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Claude OAuth Proxy (Claude Code identity)")
    logger.info(f"Token: {token_state['access_token'][:25]}...")
    logger.info(f"Refresh token: {'YES' if REFRESH_TOKEN else 'NO'}")
    logger.info(f"Endpoint: https://api.anthropic.com/v1/messages")
    logger.info(f"Beta flags: {BETA_FLAGS}")
    logger.info("=" * 60)
    logger.info("Set for RD-Agent:")
    logger.info("  OPENAI_API_BASE=http://localhost:5555/v1")
    logger.info("  OPENAI_API_KEY=dummy")
    logger.info("=" * 60)
    app.run(host="0.0.0.0", port=5555, debug=False)
