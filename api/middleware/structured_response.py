from starlette.types import ASGIApp, Receive, Scope, Send
import json
import logging

logger = logging.getLogger("uvicorn.error")

class StructuredResponseMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        # Skip non-HTTP and preflight OPTIONS requests
        if scope["type"] != "http" or scope["method"] == "OPTIONS":
            await self.app(scope, receive, send)
            return

        logger.info("📥 StructuredResponseMiddleware triggered")

        send_buffer = {}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                send_buffer["headers"] = message["headers"]
                send_buffer["status"] = message["status"]

            elif message["type"] == "http.response.body":
                body = message.get("body", b"")
                try:
                    decoded = body.decode("utf-8")
                    parsed = json.loads(decoded)
                    logger.info(f"📦 Parsed original response: {parsed}")

                    if isinstance(parsed, dict) and "responseBody" in parsed:
                        wrapped = parsed
                    else:
                        wrapped = {
                            "responseBody": parsed,
                            "responseCode": send_buffer["status"]
                        }

                except Exception as e:
                    logger.warning(f"❌ JSON parse failed: {str(e)}")
                    wrapped = {
                        "responseBody": body.decode("utf-8", errors="replace"),
                        "responseCode": send_buffer["status"]
                    }

                new_body = json.dumps(wrapped).encode("utf-8")
                logger.info(f"📤 Final wrapped response: {wrapped}")

                # ✅ Filter out Content-Length, preserve all other headers (like CORS)
                filtered_headers = [
                    (k, v) for k, v in send_buffer["headers"]
                    if k.lower() != b"content-length"
                ]
                filtered_headers.append((b"content-type", b"application/json"))

                await send({
                    "type": "http.response.start",
                    "status": send_buffer["status"],
                    "headers": filtered_headers
                })
                await send({
                    "type": "http.response.body",
                    "body": new_body,
                    "more_body": False
                })

        await self.app(scope, receive, send_wrapper)
