"""
Example client for the inference server.

Usage:
    python serving/client.py --prompt "Explain LoRA in one sentence."
    python serving/client.py --chat "What is QLoRA?" --stream
"""

import argparse
import json

import httpx


def generate(base_url: str, prompt: str, stream: bool, max_new_tokens: int) -> None:
    url = f"{base_url}/generate"
    payload = {"prompt": prompt, "max_new_tokens": max_new_tokens, "stream": stream}
    if stream:
        with httpx.stream("POST", url, json=payload, timeout=None) as r:
            for chunk in r.iter_text():
                print(chunk, end="", flush=True)
        print()
    else:
        r = httpx.post(url, json=payload, timeout=None)
        r.raise_for_status()
        print(json.dumps(r.json(), indent=2))


def chat(base_url: str, message: str, stream: bool, max_new_tokens: int) -> None:
    url = f"{base_url}/chat"
    payload = {
        "messages": [{"role": "user", "content": message}],
        "max_new_tokens": max_new_tokens,
        "stream": stream,
    }
    if stream:
        with httpx.stream("POST", url, json=payload, timeout=None) as r:
            for chunk in r.iter_text():
                print(chunk, end="", flush=True)
        print()
    else:
        r = httpx.post(url, json=payload, timeout=None)
        r.raise_for_status()
        print(json.dumps(r.json(), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Client for the LLM inference server.")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--prompt", help="Send a raw completion prompt.")
    parser.add_argument("--chat", help="Send a single chat message.")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    if args.chat:
        chat(args.url, args.chat, args.stream, args.max_new_tokens)
    elif args.prompt:
        generate(args.url, args.prompt, args.stream, args.max_new_tokens)
    else:
        parser.error("Provide --prompt or --chat.")


if __name__ == "__main__":
    main()
