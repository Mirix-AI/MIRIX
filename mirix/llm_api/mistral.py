import httpx

from mirix.utils import printd, smart_urljoin


async def mistral_get_model_list(url: str, api_key: str) -> dict:
    url = smart_urljoin(url, "models")

    headers = {"Content-Type": "application/json"}
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"

    printd(f"Sending request to {url}")
    response = None
    try:
        # TODO add query param "tool" to be true
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
        response.raise_for_status()  # Raises HTTPStatusError for 4XX/5XX status
        response_json = response.json()  # convert to dict from string
        return response_json
    except httpx.HTTPStatusError as http_err:
        # Handle HTTP errors (e.g., response 4XX, 5XX)
        try:
            if response:
                response = response.json()
        except Exception:
            pass
        printd(f"Got HTTPError, exception={http_err}, response={response}")
        raise http_err
    except httpx.RequestError as req_err:
        # Handle other requests-related errors (e.g., connection error)
        try:
            if response:
                response = response.json()
        except Exception:
            pass
        printd(f"Got RequestException, exception={req_err}, response={response}")
        raise req_err
    except Exception as e:
        # Handle other potential errors
        try:
            if response:
                response = response.json()
        except Exception:
            pass
        printd(f"Got unknown Exception, exception={e}, response={response}")
        raise e
