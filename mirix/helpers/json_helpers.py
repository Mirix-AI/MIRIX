import json
from datetime import datetime

import demjson3 as demjson

from mirix.log import get_logger

logger = get_logger(__name__)


def json_loads(data):
    return json.loads(data, strict=False)


def json_dumps(data, indent=2):
    def safe_serializer(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")

    return json.dumps(data, indent=indent, default=safe_serializer, ensure_ascii=False)


def parse_json(string) -> dict:
    """Parse JSON string into JSON with both json and demjson"""
    result = None
    try:
        result = json_loads(string)
        return result
    except Exception as e:
        logger.debug("Error parsing json with json package: %s", e)

    try:
        result = demjson.decode(string)
        return result
    except demjson.JSONDecodeError as e:
        logger.debug("Error parsing json with demjson package: %s", e)

    try:
        from json_repair import repair_json

        string = repair_json(string)
        result = json_loads(string)
        return result

    except Exception as e:
        logger.debug("Error repairing json with json_repair package: %s", e)
        raise e
