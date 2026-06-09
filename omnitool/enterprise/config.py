"""
Configuration management for Omnitool Enterprise.
Loads settings from environment variables and .env files.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class GemmaConfig:
    """Configuration for the Gemma 4 model."""
    base_url: str = os.getenv("GEMMA_BASE_URL", "http://localhost:8080/v1")
    api_key: str = os.getenv("GEMMA_API_KEY", "not-needed")
    model: str = os.getenv("GEMMA_MODEL", "gemma-4")
    temperature: float = float(os.getenv("GEMMA_TEMPERATURE", "0.3"))
    max_tokens: int = int(os.getenv("GEMMA_MAX_TOKENS", "4096"))


@dataclass
class SlackConfig:
    """Configuration for Slack integration."""
    bot_token: str = os.getenv("SLACK_BOT_TOKEN", "")
    app_token: str = os.getenv("SLACK_APP_TOKEN", "")
    signing_secret: str = os.getenv("SLACK_SIGNING_SECRET", "")


@dataclass
class AzureConfig:
    """Configuration for Azure operations."""
    subscription_id: str = os.getenv("AZURE_SUBSCRIPTION_ID", "")
    resource_group: str = os.getenv("AZURE_RESOURCE_GROUP", "")
    tenant_id: str = os.getenv("AZURE_TENANT_ID", "")
    client_id: str = os.getenv("AZURE_CLIENT_ID", "")
    client_secret: str = os.getenv("AZURE_CLIENT_SECRET", "")


@dataclass
class ServiceNowConfig:
    """Configuration for ServiceNow integration."""
    instance_url: str = os.getenv("SNOW_INSTANCE_URL", "")
    username: str = os.getenv("SNOW_USERNAME", "")
    password: str = os.getenv("SNOW_PASSWORD", "")


@dataclass
class EmailConfig:
    """Configuration for email operations."""
    smtp_server: str = os.getenv("EMAIL_SMTP_SERVER", "")
    smtp_port: int = int(os.getenv("EMAIL_SMTP_PORT", "587"))
    imap_server: str = os.getenv("EMAIL_IMAP_SERVER", "")
    username: str = os.getenv("EMAIL_USERNAME", "")
    password: str = os.getenv("EMAIL_PASSWORD", "")


@dataclass
class OmniParserConfig:
    """Configuration for OmniParser server."""
    server_url: str = os.getenv("OMNIPARSER_SERVER_URL", "http://localhost:8000")


@dataclass
class MemoryConfig:
    """Configuration for the memory system."""
    db_path: str = os.getenv("MEMORY_DB_PATH", "omnitool_memory.db")
    max_context_turns: int = int(os.getenv("MEMORY_MAX_CONTEXT_TURNS", "20"))


@dataclass
class EnterpriseConfig:
    """Root configuration aggregating all subsystem configs."""
    gemma: GemmaConfig = field(default_factory=GemmaConfig)
    slack: SlackConfig = field(default_factory=SlackConfig)
    azure: AzureConfig = field(default_factory=AzureConfig)
    servicenow: ServiceNowConfig = field(default_factory=ServiceNowConfig)
    email: EmailConfig = field(default_factory=EmailConfig)
    omniparser: OmniParserConfig = field(default_factory=OmniParserConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    host_control_url: str = os.getenv("HOST_CONTROL_URL", "http://localhost:5000")
    approval_timeout_seconds: int = int(os.getenv("APPROVAL_TIMEOUT_SECONDS", "300"))
    debug: bool = os.getenv("OMNITOOL_DEBUG", "false").lower() == "true"


# Singleton config instance
_config: Optional[EnterpriseConfig] = None


def get_config() -> EnterpriseConfig:
    global _config
    if _config is None:
        _config = EnterpriseConfig()
    return _config
