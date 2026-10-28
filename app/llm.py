import os, time, logging
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from google import genai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OLLAMA_URL = os.environ["OLLAMA_URL"]
EMBED_MODEL = os.environ["EMBED_MODEL"]

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
def embed(text: str) -> list[float]:
    start = time.perf_counter()
    response = httpx.post(
        f"{OLLAMA_URL}/api/embeddings",
        json = {"model": EMBED_MODEL, "prompt": text},
        timeout = 30
    )
    response.raise_for_status()
    vector = response.json()["embedding"]
    ms = (time.perf_counter() - start) * 1000
    logger.info("embed took %.0f ms", ms)
    return vector

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
def generate(prompt: str) -> str:
    start = time.perf_counter()
    response = client.models.generate_content(
        model = os.environ["GEMINI_MODEL"],
        contents = prompt,
    )
    ms = (time.perf_counter() - start) * 1000
    tokens = response.usage_metadata.total_token_count
    logger.info("generate took %.0f ms, %s tokens", ms, tokens)
    return response.text