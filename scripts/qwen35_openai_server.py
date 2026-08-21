import argparse
import json
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.llm_backend import generate_chat_text, load_llm_backend


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal OpenAI-compatible server for local Qwen3.5 smoke tests.")
    parser.add_argument("--qwen_ckpt", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--max_tokens_limit", type=int, default=None,
                        help="Optional cap for max_tokens, useful for slow CPU smoke tests.")
    return parser.parse_args()


def make_handler(model, processor, model_name, max_tokens_limit=None):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self._send_json({"status": "ok", "model": model_name})
                return
            self.send_error(404, "not found")

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.send_error(404, "not found")
                return

            try:
                length = int(self.headers.get("content-length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                messages = body["messages"]
                max_tokens = int(body.get("max_tokens", 256))
                if max_tokens_limit is not None:
                    max_tokens = min(max_tokens, max_tokens_limit)
                temperature = body.get("temperature")
                top_p = body.get("top_p")
                seed = body.get("seed")
                do_sample = None
                if temperature is not None:
                    do_sample = float(temperature) > 0
                text = generate_chat_text(
                    model,
                    processor,
                    messages,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    top_p=top_p,
                    seed=seed,
                )
                self._send_json({
                    "id": "pathagent-qwen35-local",
                    "object": "chat.completion",
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": "stop",
                        }
                    ],
                })
            except Exception as exc:
                traceback.print_exc()
                self._send_json({"error": {"message": str(exc), "type": type(exc).__name__}}, status=500)

        def _send_json(self, payload, status=200):
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, fmt, *args):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    return Handler


def main():
    args = parse_args()
    model_name = args.model_name or args.qwen_ckpt
    model, processor = load_llm_backend(args.qwen_ckpt, backend="transformers")
    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(model, processor, model_name, args.max_tokens_limit),
    )
    print(f"serving {model_name} at http://{args.host}:{args.port}/v1", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
