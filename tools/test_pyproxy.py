"""Quick verification script for the rotating proxy configuration.

PowerShell usage with an activated virtualenv (EXAMPLE — do not commit):

    set PYPROXY_ENABLED=true
    set PYPROXY_HOST=925559a762982876.zqq.na.pyproxy.io
    set PYPROXY_PORT=16666
    set PYPROXY_USERNAME=deuxt921-zone-resi
    set PYPROXY_PASSWORD=deuxt129
    python tools/test_pyproxy.py
"""
from __future__ import annotations

import asyncio
import json
import logging

import aiohttp
import requests

from proxy_manager import ProxyManager

IPIFY_URL = "https://api.ipify.org?format=json"


def test_requests_proxy(manager: ProxyManager) -> None:
    """Validate requests-based proxy routing."""

    proxies = manager.as_requests()
    if not proxies:
        logging.info("Proxy disabled — skipping requests test")
        return

    logging.info("Testing requests proxy against %s", IPIFY_URL)
    response = requests.get(IPIFY_URL, proxies=proxies, timeout=15)
    response.raise_for_status()
    payload = response.json()
    logging.info("requests reported IP: %s", payload.get("ip"))
    print("requests ->", json.dumps(payload))


async def test_aiohttp_proxy(manager: ProxyManager) -> None:
    """Validate aiohttp proxy routing."""

    proxy_url = manager.as_aiohttp()
    if not proxy_url:
        logging.info("Proxy disabled — skipping aiohttp test")
        return

    logging.info("Testing aiohttp proxy against %s", IPIFY_URL)
    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(IPIFY_URL, proxy=proxy_url) as response:
            response.raise_for_status()
            payload = await response.json()
            logging.info("aiohttp reported IP: %s", payload.get("ip"))
            print("aiohttp ->", json.dumps(payload))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    manager = ProxyManager()
    masked = manager.masked_proxy()
    if masked:
        logging.info("Using proxy %s", masked)
    else:
        logging.info("Proxy disabled — requests will hit the direct connection")

    try:
        test_requests_proxy(manager)
    except requests.RequestException as exc:
        logging.error("requests proxy test failed: %s", exc, exc_info=True)
        return

    try:
        asyncio.run(test_aiohttp_proxy(manager))
    except aiohttp.ClientError as exc:
        logging.error("aiohttp proxy test failed: %s", exc, exc_info=True)


if __name__ == "__main__":
    main()
