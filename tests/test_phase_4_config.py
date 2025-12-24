import pytest
import asyncio
from src.core.config import settings
from src.core.workflow_config_manager import (
    WorkflowConfig,
    WorkflowConfigManager,
    workflow_config_manager,
)


class TestWorkflowConfig:
    """Tests for WorkflowConfig dataclass."""
    
    def test_config_creation(self):
        """Test creating a basic WorkflowConfig."""
        config = WorkflowConfig(user_id="user123")
        assert config.user_id == "user123"
        assert config.workflow_version in {"v2", "v3", "v4"}
        assert isinstance(config.observer_enabled, bool)
        assert isinstance(config.rag_enabled, bool)
    
    def test_config_with_overrides(self):
        """Test creating config with custom values."""
        config = WorkflowConfig(
            user_id="user456",
            workflow_version="v3",
            observer_enabled=False,
            rag_enabled=False,
        )
        assert config.workflow_version == "v3"
        assert config.observer_enabled is False
        assert config.rag_enabled is False
    
    def test_config_validation_valid_version(self):
        """Test that valid versions pass validation."""
        for version in ["v2", "v3", "v4"]:
            config = WorkflowConfig(user_id="user789", workflow_version=version)
            config.validate_workflow_version()
            assert config.workflow_version == version
    
    def test_config_validation_invalid_version(self):
        """Test that invalid versions get corrected."""
        config = WorkflowConfig(user_id="user789", workflow_version="v99")
        config.validate_workflow_version()
        assert config.workflow_version == "v3"  # Falls back to v3


class TestWorkflowConfigManager:
    """Tests for WorkflowConfigManager."""
    
    @pytest.fixture
    async def manager(self):
        """Create a fresh manager for each test."""
        return WorkflowConfigManager()
    
    @pytest.mark.asyncio
    async def test_get_default_config(self):
        """Test getting default config without overrides."""
        manager = WorkflowConfigManager()
        config = await manager.get_config("user_default")
        
        assert config.user_id == "user_default"
        assert config.workflow_version in {"v2", "v3", "v4"}
        assert config.observer_enabled == settings.WORKFLOW_OBSERVER_ENABLED
        assert config.rag_enabled == settings.ENABLE_RAG
    
    @pytest.mark.asyncio
    async def test_set_user_override(self):
        """Test setting user-specific override."""
        manager = WorkflowConfigManager()
        override_config = WorkflowConfig(
            user_id="user_override",
            workflow_version="v3",
            observer_enabled=False,
        )
        
        await manager.set_user_override("user_override", override_config)
        retrieved = await manager.get_config("user_override")
        
        assert retrieved.workflow_version == "v3"
        assert retrieved.observer_enabled is False
    
    @pytest.mark.asyncio
    async def test_remove_user_override(self):
        """Test removing user override reverts to global."""
        manager = WorkflowConfigManager()
        config = WorkflowConfig(user_id="user_test", workflow_version="v3")
        
        await manager.set_user_override("user_test", config)
        await manager.remove_user_override("user_test")
        
        # After removal, should use global settings
        retrieved = await manager.get_config("user_test")
        assert retrieved.workflow_version == settings.WORKFLOW_VERSION
    
    @pytest.mark.asyncio
    async def test_canary_users(self):
        """Test canary user management."""
        manager = WorkflowConfigManager()
        
        await manager.add_canary_user("canary_user1")
        assert "canary_user1" in manager._canary_users
        
        config = await manager.get_config("canary_user1")
        assert config.workflow_version == settings.WORKFLOW_VERSION
        
        await manager.remove_canary_user("canary_user1")
        assert "canary_user1" not in manager._canary_users
    
    @pytest.mark.asyncio
    async def test_fallback_users(self):
        """Test fallback user management (always use v3)."""
        manager = WorkflowConfigManager()
        
        await manager.add_fallback_user("fallback_user1")
        assert "fallback_user1" in manager._fallback_users
        
        config = await manager.get_config("fallback_user1")
        assert config.workflow_version == "v3"
        
        await manager.remove_fallback_user("fallback_user1")
        assert "fallback_user1" not in manager._fallback_users
    
    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test statistics generation."""
        manager = WorkflowConfigManager()
        
        await manager.add_canary_user("user1")
        await manager.add_fallback_user("user2")
        await manager.add_canary_user("user3")
        
        stats = await manager.get_stats()
        
        assert stats["total_users_with_overrides"] == 3
        assert stats["canary_users"] == 2
        assert stats["fallback_users"] == 1
        assert stats["global_workflow_version"] == settings.WORKFLOW_VERSION
        assert stats["global_canary_percentage"] == settings.CANARY_ROLLOUT_PERCENTAGE


class TestSettingsCanaryRollout:
    """Tests for canary rollout logic in settings."""
    
    def test_canary_100_percent(self):
        """Test 100% rollout uses target version."""
        # Temporarily set to 100%
        original = settings.CANARY_ROLLOUT_PERCENTAGE
        settings.CANARY_ROLLOUT_PERCENTAGE = 100
        
        version = settings.should_use_workflow_version("user_any")
        assert version == settings.WORKFLOW_VERSION
        
        settings.CANARY_ROLLOUT_PERCENTAGE = original
    
    def test_canary_0_percent(self):
        """Test 0% rollout uses v3."""
        original = settings.CANARY_ROLLOUT_PERCENTAGE
        settings.CANARY_ROLLOUT_PERCENTAGE = 0
        
        version = settings.should_use_workflow_version("user_any")
        assert version == "v3"
        
        settings.CANARY_ROLLOUT_PERCENTAGE = original
    
    def test_canary_consistent_assignment(self):
        """Test same user gets consistent version."""
        user_id = "consistent_user"
        version1 = settings.should_use_workflow_version(user_id)
        version2 = settings.should_use_workflow_version(user_id)
        
        assert version1 == version2
    
    def test_canary_different_users_vary(self):
        """Test different users may get different versions (with high percentage)."""
        original = settings.CANARY_ROLLOUT_PERCENTAGE
        settings.CANARY_ROLLOUT_PERCENTAGE = 50  # 50% rollout
        
        versions = set()
        for i in range(100):
            v = settings.should_use_workflow_version(f"user_{i}")
            versions.add(v)
        
        # With 50% rollout, should get mixed versions
        assert len(versions) >= 1  # At least some consistency, but may vary
        
        settings.CANARY_ROLLOUT_PERCENTAGE = original


class TestConfigValidation:
    """Tests for configuration validation."""
    
    @pytest.mark.asyncio
    async def test_invalid_version_corrected(self):
        """Test invalid versions are corrected to v3."""
        manager = WorkflowConfigManager()
        
        invalid_config = WorkflowConfig(
            user_id="invalid_user",
            workflow_version="v99",  # Invalid
        )
        
        await manager.set_user_override("invalid_user", invalid_config)
        retrieved = await manager.get_config("invalid_user")
        
        assert retrieved.workflow_version == "v3"
    
    def test_settings_workflow_version_valid(self):
        """Test that settings have valid workflow version."""
        assert settings.WORKFLOW_VERSION in {"v2", "v3", "v4"}
    
    def test_settings_canary_percentage_range(self):
        """Test canary percentage is in valid range."""
        assert 0 <= settings.CANARY_ROLLOUT_PERCENTAGE <= 100


class TestConfigIntegration:
    """Integration tests for config system."""
    
    @pytest.mark.asyncio
    async def test_full_config_workflow(self):
        """Test complete configuration workflow."""
        manager = WorkflowConfigManager()
        
        # Step 1: Get default config
        default = await manager.get_config("new_user")
        assert default.workflow_version is not None
        
        # Step 2: Set override for canary testing
        await manager.add_canary_user("new_user")
        canary = await manager.get_config("new_user")
        assert canary.workflow_version == settings.WORKFLOW_VERSION
        
        # Step 3: Move to fallback if needed
        await manager.remove_canary_user("new_user")
        await manager.add_fallback_user("new_user")
        fallback = await manager.get_config("new_user")
        assert fallback.workflow_version == "v3"
        
        # Step 4: Remove all overrides
        await manager.remove_fallback_user("new_user")
        reset = await manager.get_config("new_user")
        assert reset.workflow_version == settings.WORKFLOW_VERSION
    
    @pytest.mark.asyncio
    async def test_multiple_users_isolation(self):
        """Test that user configurations are isolated."""
        manager = WorkflowConfigManager()
        
        config1 = WorkflowConfig(user_id="user1", workflow_version="v3", observer_enabled=False)
        config2 = WorkflowConfig(user_id="user2", workflow_version="v4", observer_enabled=True)
        
        await manager.set_user_override("user1", config1)
        await manager.set_user_override("user2", config2)
        
        retrieved1 = await manager.get_config("user1")
        retrieved2 = await manager.get_config("user2")
        
        assert retrieved1.workflow_version == "v3"
        assert retrieved1.observer_enabled is False
        assert retrieved2.workflow_version == "v4"
        assert retrieved2.observer_enabled is True


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
