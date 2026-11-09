"""Numeric tokens that must survive memory and graph canonicalization."""

import re

# Keep signed values, currencies, decimals, ranges, dates, ratios, exponents and
# percentages whole. Splitting "1.2" into the set {"1", "2"} loses its value.
NUMBER_PATTERN = (
    r"(?<!\d)[+\-\u2212]?(?:[$€£¥][+\-\u2212]?)?"
    r"(?:\d+(?:[.,:/-]\d+)*|\.\d+)(?:[eE][+-]?\d+)?%?"
    r"(?:[a-zA-Z]+(?:/[a-zA-Z]+)?|[ ]+(?:usd|eur|gbp|jpy|cny|kg|mg|km|cm|mm|ml|lbs?|oz|g|m|l)\b)?"
)
NUMBER_TOKEN = re.compile(NUMBER_PATTERN, re.I)
