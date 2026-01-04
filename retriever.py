# retriever.py
import os
import json
import logging
from dotenv import load_dotenv
from openai import OpenAI

# optional: requests may not be in venv by default — typically installed
import requests

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "olist_docs")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set in .env")

# OpenAI client (SDK v1+)
openai_client = OpenAI(api_key=OPENAI_API_KEY)


def get_embedding(text: str):
    """Return embedding vector (list[float]) for given text."""
    resp = openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    # Defensive: different sdk shapes, but usually resp.data[0].embedding
    try:
        return resp.data[0].embedding
    except Exception:
        # try alternative dict-style
        return resp["data"][0]["embedding"]


def _search_via_http(vector, top_k: int = 5):
    """
    Call Qdrant HTTP API /collections/{name}/points/search
    Works regardless of qdrant-client python lib version.
    """
    url = QDRANT_URL.rstrip("/") + f"/collections/{QDRANT_COLLECTION}/points/search"
    headers = {"Content-Type": "application/json"}
    if QDRANT_API_KEY:
        # Qdrant cloud / service often expects 'api-key' header
        headers["api-key"] = QDRANT_API_KEY

    body = {
        "vector": vector,
        "limit": top_k,
        "with_payload": True
    }

    logger.debug("POST %s %s", url, json.dumps(body)[:1000])
    resp = requests.post(url, headers=headers, json=body, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    # Qdrant REST usually returns {"result": [...]} or {"result": {"points": [...]}}
    if isinstance(data, dict):
        # common shape: {"result": [ {id, payload, score}, ... ] }
        if "result" in data and isinstance(data["result"], list):
            return data["result"]
        # alternative: {"result": {"points": [...]}}
        if "result" in data and isinstance(data["result"], dict) and "points" in data["result"]:
            return data["result"]["points"]
        # sometimes top-level "points"
        if "points" in data and isinstance(data["points"], list):
            return data["points"]
    # fallback - if response itself is a list
    if isinstance(data, list):
        return data

    # nothing matched
    raise RuntimeError("Unexpected Qdrant HTTP response shape: " + str(data)[:1000])


def _try_qdrant_client_fallback(qdrant, vector, top_k: int = 5):
    """
    Try a variety of possible qdrant-client method names and return results in consistent list form.
    This is only used if HTTP fallback is not desired/possible.
    """
    attempts = [
        ("qdrant.search", lambda obj: getattr(obj, "search")(collection_name=QDRANT_COLLECTION, query=vector, limit=top_k)),
        ("qdrant.search_with_query_vector", lambda obj: getattr(obj, "search")(collection_name=QDRANT_COLLECTION, query_vector=vector, limit=top_k)),
        ("qdrant.search_points", lambda obj: getattr(obj, "search_points")(collection_name=QDRANT_COLLECTION, query={"vector": vector, "limit": top_k})),
        ("qdrant.points.search", lambda obj: getattr(obj.points, "search")(collection_name=QDRANT_COLLECTION, query=vector, limit=top_k)),
        ("qdrant.http.points_api.search", lambda obj: obj.http.points_api.search(collection_name=QDRANT_COLLECTION, search_request={"vector": vector, "limit": top_k})),
        ("qdrant.http.points_api.search_request_models", lambda obj: obj.http.points_api.search(collection_name=QDRANT_COLLECTION, search_request={"vector": vector, "limit": top_k})),
    ]

    for name, fn in attempts:
        try:
            logger.debug("Attempting qdrant client method: %s", name)
            res = fn(qdrant)
            # Normalize a few possible return shapes:
            # - list of points
            # - object with .result or .points attributes
            if isinstance(res, list):
                return res
            if hasattr(res, "result"):
                maybe = getattr(res, "result")
                if isinstance(maybe, list):
                    return maybe
                if hasattr(maybe, "points"):
                    return maybe.points
            if hasattr(res, "points"):
                return res.points
            # if dict
            if isinstance(res, dict) and "result" in res:
                return res["result"]
            # else try treating as iterable
            try:
                return list(res)
            except Exception:
                pass
        except AttributeError:
            continue
        except Exception as e:
            logger.warning("qdrant client attempt %s failed: %s", name, e)
            continue

    raise RuntimeError("No usable qdrant client search method found.")


def retrieve(query: str, top_k: int = 5, qdrant_client=None):
    """
    Retrieve top_k docs from Qdrant for the query.
    By default this implementation uses direct HTTP to Qdrant REST API which is version agnostic.
    If 'qdrant_client' is provided (an instance of QdrantClient), it will be used as fallback attempt.
    """
    qvec = get_embedding(query)

    # Preferred: use HTTP REST endpoint (most robust)
    try:
        logger.info("Searching Qdrant via HTTP REST endpoint...")
        hits = _search_via_http(qvec, top_k=top_k)
    except Exception as e_http:
        logger.warning("HTTP search failed: %s - will try qdrant-client methods if provided.", e_http)
        if qdrant_client is None:
            # try to import qdrant client on the fly
            try:
                from qdrant_client import QdrantClient as _QC
                qdrant_client = _QC(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            except Exception as e:
                raise RuntimeError("HTTP search failed and qdrant-client not available: " + str(e_http)) from e_http

        # try multiple qdrant client ways
        hits = _try_qdrant_client_fallback(qdrant_client, qvec, top_k=top_k)

    # Normalize hit objects to standard dicts with id, score, payload
    normalized = []
    for h in hits:
        # h could be dict-like or object-like
        try:
            if isinstance(h, dict):
                _id = h.get("id") or h.get("point", {}).get("id")
                payload = h.get("payload") or h.get("point", {}).get("payload") or h.get("payload", {})
                score = h.get("score") or h.get("dist") or h.get("distance")
            else:
                # dot-attribute objects from qdrant-client
                _id = getattr(h, "id", None) or getattr(h, "point", None) and getattr(h.point, "id", None)
                payload = getattr(h, "payload", None) or getattr(h, "point", None) and getattr(h.point, "payload", None)
                score = getattr(h, "score", None) or getattr(h, "distance", None)
            normalized.append({
                "id": _id,
                "score": score,
                "payload": payload or {}
            })
        except Exception as e:
            logger.warning("Failed to normalize hit: %s -- raw: %s", e, str(h)[:400])
            continue

    return normalized
