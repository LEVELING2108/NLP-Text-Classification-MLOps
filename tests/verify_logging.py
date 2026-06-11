from mlops_nlp.logging_config import configure_logging, get_logger
import json

def test_logging():
    configure_logging("INFO")
    logger = get_logger("test_logger")
    
    # We'll use a string buffer or just check the output manually
    # For automated check, we can use capsys in pytest, but here we'll just run it.
    logger.info("This is a test message")
    logger.info("Message with extra context", extra={"request_id": "12345", "user_id": "abc"})
    
    try:
        1/0
    except Exception:
        logger.exception("An error occurred")

if __name__ == "__main__":
    test_logging()
