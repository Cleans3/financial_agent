import logging
from typing import Dict, Optional, Set
from dataclasses import dataclass, field
from src.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class WorkflowConfig:
    """Per-user workflow configuration with overrides."""
    user_id: str
    workflow_version: str = field(default_factory=lambda: settings.WORKFLOW_VERSION)
    observer_enabled: bool = field(default_factory=lambda: settings.WORKFLOW_OBSERVER_ENABLED)
    rag_enabled: bool = field(default_factory=lambda: settings.ENABLE_RAG)
    tools_enabled: bool = field(default_factory=lambda: settings.ENABLE_TOOLS)
    query_rewriting_enabled: bool = field(default_factory=lambda: settings.ENABLE_QUERY_REWRITING)
    
    def validate_workflow_version(self) -> None:
        """Validate workflow version is valid."""
        valid_versions = {"v2", "v3", "v4"}
        if self.workflow_version not in valid_versions:
            logger.warning(f"Invalid workflow version {self.workflow_version}, using v3")
            self.workflow_version = "v3"


class WorkflowConfigManager:
    """Manages workflow configuration with per-user overrides."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._user_overrides: Dict[str, WorkflowConfig] = {}
        self._canary_users: Set[str] = set()
        self._fallback_users: Set[str] = set()
    
    async def get_config(self, user_id: str) -> WorkflowConfig:
        """Get workflow configuration for user.
        
        Returns per-user config if override exists, otherwise generates
        from global settings with canary rollout logic.
        """
        # Check if user has explicit override
        if user_id in self._user_overrides:
            return self._user_overrides[user_id]
        
        # Create config from global settings with canary logic
        config = WorkflowConfig(
            user_id=user_id,
            workflow_version=settings.should_use_workflow_version(user_id),
            observer_enabled=settings.WORKFLOW_OBSERVER_ENABLED,
            rag_enabled=settings.ENABLE_RAG,
            tools_enabled=settings.ENABLE_TOOLS,
            query_rewriting_enabled=settings.ENABLE_QUERY_REWRITING,
        )
        config.validate_workflow_version()
        return config
    
    async def set_user_override(self, user_id: str, config: WorkflowConfig) -> None:
        """Set explicit workflow configuration override for user."""
        config.validate_workflow_version()
        self._user_overrides[user_id] = config
        self.logger.info(f"Set workflow override for user {user_id}: {config.workflow_version}")
    
    async def remove_user_override(self, user_id: str) -> None:
        """Remove explicit override, revert to global settings."""
        if user_id in self._user_overrides:
            del self._user_overrides[user_id]
            self.logger.info(f"Removed workflow override for user {user_id}")
    
    async def add_canary_user(self, user_id: str) -> None:
        """Add user to canary rollout group."""
        self._canary_users.add(user_id)
        config = await self.get_config(user_id)
        await self.set_user_override(user_id, config)
        self.logger.info(f"Added user {user_id} to canary rollout")
    
    async def remove_canary_user(self, user_id: str) -> None:
        """Remove user from canary rollout group."""
        if user_id in self._canary_users:
            self._canary_users.remove(user_id)
            await self.remove_user_override(user_id)
            self.logger.info(f"Removed user {user_id} from canary rollout")
    
    async def add_fallback_user(self, user_id: str) -> None:
        """Add user to fallback group (always use v3)."""
        self._fallback_users.add(user_id)
        config = await self.get_config(user_id)
        config.workflow_version = "v3"
        config.validate_workflow_version()
        await self.set_user_override(user_id, config)
        self.logger.info(f"Added user {user_id} to fallback group (v3)")
    
    async def remove_fallback_user(self, user_id: str) -> None:
        """Remove user from fallback group."""
        if user_id in self._fallback_users:
            self._fallback_users.remove(user_id)
            await self.remove_user_override(user_id)
            self.logger.info(f"Removed user {user_id} from fallback group")
    
    async def get_stats(self) -> Dict:
        """Get workflow configuration statistics."""
        return {
            "total_users_with_overrides": len(self._user_overrides),
            "canary_users": len(self._canary_users),
            "fallback_users": len(self._fallback_users),
            "global_workflow_version": settings.WORKFLOW_VERSION,
            "global_canary_percentage": settings.CANARY_ROLLOUT_PERCENTAGE,
            "observer_enabled": settings.WORKFLOW_OBSERVER_ENABLED,
        }


# Global instance
workflow_config_manager = WorkflowConfigManager()
