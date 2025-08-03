import time
import logging
import json
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("setpoint-log")


# Create a custom middleware to log the incoming requests
class LogRequestMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Log request method and URL
        logger.info(f"Incoming request: {request.method} {request.url}")

        # Read the content type to handle binary data differently
        content_type = request.headers.get("content-type", "")

        if "multipart/form-data" in content_type:
            logger.info(
                "Request contains multipart/form-data (likely a file upload), skipping body logging."
            )
        else:
            # Read and log the request body
            body = await request.body()

            if body:
                try:
                    json_body = json.loads(body.decode("utf-8"))
                    logger.info(f"Request payload: {json.dumps(json_body, indent=2)}")
                except (json.JSONDecodeError, UnicodeDecodeError):
                    logger.info(
                        f"Request payload: {body.decode('utf-8', errors='ignore')}"
                    )  # Handle decoding errors safely
            else:
                logger.info("Request payload: No Body")

        # Track processing time
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(f"Completed request in {process_time:.2f} seconds")

        return response
