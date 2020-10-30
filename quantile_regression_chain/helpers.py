from collections import defaultdict

import logging
logger = logging.getLogger(__name__)


def rec_dd():
    return defaultdict(rec_dd)

def scan_dict(dic):
    for key, value in dic.items():
        if isinstance(value, dict):
            yield from scan_dict(value)
        else:
            yield value
