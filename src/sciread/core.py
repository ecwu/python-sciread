from .logging_config import get_logger

logger = get_logger(__name__)


def compute(args):
    """Compute function that returns the longest string from input arguments."""
    logger.debug(f"Computing result for {len(args)} arguments: {args}")

    if not args:
        logger.warning("No arguments provided to compute function")
        return ""

    result = max(args, key=len)
    logger.debug(f"Compute result: {result}")
    return result
