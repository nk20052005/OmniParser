import os
from pathlib import Path

from openai import OpenAI

from .utils import encode_image, is_image_path


def load_dotenv(dotenv_path: Path) -> None:
	if not dotenv_path.exists():
		return

	for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
		line = raw_line.strip()
		if not line or line.startswith("#") or "=" not in line:
			continue

		key, value = line.split("=", 1)
		key = key.strip()
		value = value.strip().strip('"').strip("'")
		if key and key not in os.environ:
			os.environ[key] = value


load_dotenv(Path(__file__).resolve().parents[4] / ".env")


def _build_messages(messages: list | str, system: str, model_name: str) -> list:
	final_messages = [{"role": "system", "content": system}]

	if isinstance(messages, str):
		return [{"role": "user", "content": messages}]

	for item in messages:
		contents = []
		if isinstance(item, dict):
			for cnt in item.get("content", []):
				if isinstance(cnt, str):
					if is_image_path(cnt) and "o3-mini" not in model_name:
						base64_image = encode_image(cnt)
						content = {
							"type": "image_url",
							"image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
						}
					else:
						content = {"type": "text", "text": cnt}
				else:
					content = {"type": "text", "text": str(cnt)}
				contents.append(content)
			message = {"role": "user", "content": contents}
		else:
			contents.append({"type": "text", "text": str(item)})
			message = {"role": "user", "content": contents}
		final_messages.append(message)

	return final_messages


def run_oai_interleaved(
	messages: list,
	system: str,
	model_name: str,
	api_key: str,
	max_tokens: int = 256,
	temperature: float = 0,
	provider_base_url: str = "https://api.openai.com/v1",
):
	endpoint = os.getenv("OPENAI_BASE_URL") or provider_base_url
	deployment_name = os.getenv("OPENAI_MODEL") or model_name

	client = OpenAI(
		base_url=endpoint,
		api_key=api_key or os.getenv("OPENAI_API_KEY"),
	)

	final_messages = _build_messages(messages, system, model_name)

	request_args = {
		"model": deployment_name,
		"messages": final_messages,
	}

	request_args["temperature"] = temperature

	try:
		completion = client.chat.completions.create(**request_args)
		text = completion.choices[0].message.content
		token_usage = int(getattr(completion.usage, "total_tokens", 0) or 0)
		if text is None:
			raise RuntimeError("LLM response had empty content")
		return text, token_usage
	except Exception as e:
		raise RuntimeError(f"Error in OpenAI/Azure request: {e}") from e