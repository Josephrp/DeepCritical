"""
Comprehensive tests for LLM model implementations.

Tests cover:
- Loading from actual config files (configs/llm/)
- Error handling (invalid inputs)
- Edge cases (boundary values)
- Configuration precedence
- Datatype validation
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError

from DeepResearch.src.datatypes.llm_models import (
    GenerationConfig,
    LLMModelConfig,
    LLMProvider,
)
from DeepResearch.src.models import LlamaCppModel, OpenAICompatibleModel, VLLMModel

# Path to config files
CONFIGS_DIR = Path(__file__).parent.parent / "configs" / "llm"


class TestOpenAICompatibleModelWithConfigs:
    """Test model creation using actual config files."""

    def test_from_vllm_with_actual_config_file(self):
        """Test loading vLLM model from actual vllm_pydantic.yaml config."""
        config_path = CONFIGS_DIR / "vllm_pydantic.yaml"
        config = OmegaConf.load(config_path)

        # Ensure config is a DictConfig (not ListConfig)
        assert OmegaConf.is_dict(config), "Config is not a dict config"
        # Cast to DictConfig for type safety
        dict_config: DictConfig = config  # type: ignore

        model = OpenAICompatibleModel.from_vllm(config=dict_config)

        # Values from vllm_pydantic.yaml
        assert model.model_name == "meta-llama/Llama-3-8B"
        assert "localhost:8000" in model.base_url

    def test_from_llamacpp_with_actual_config_file(self):
        """Test loading llama.cpp model from actual llamacpp_local.yaml config."""
        config_path = CONFIGS_DIR / "llamacpp_local.yaml"
        config = OmegaConf.load(config_path)

        # Ensure config is a DictConfig (not ListConfig)
        assert OmegaConf.is_dict(config), "Config is not a dict config"
        # Cast to DictConfig for type safety
        dict_config: DictConfig = config  # type: ignore

        model = OpenAICompatibleModel.from_llamacpp(config=dict_config)

        # Values from llamacpp_local.yaml
        assert model.model_name == "llama"
        assert "localhost:8080" in model.base_url

    def test_from_tgi_with_actual_config_file(self):
        """Test loading TGI model from actual tgi_local.yaml config."""
        config_path = CONFIGS_DIR / "tgi_local.yaml"
        config = OmegaConf.load(config_path)

        # Ensure config is a DictConfig (not ListConfig)
        assert OmegaConf.is_dict(config), "Config is not a dict config"
        # Cast to DictConfig for type safety
        dict_config: DictConfig = config  # type: ignore

        model = OpenAICompatibleModel.from_tgi(config=dict_config)

        # Values from tgi_local.yaml
        assert model.model_name == "bigscience/bloom-560m"
        assert "localhost:3000" in model.base_url

    def test_config_files_have_valid_generation_params(self):
        """Test that all config files have valid generation parameters."""
        for config_file in [
            "vllm_pydantic.yaml",
            "llamacpp_local.yaml",
            "tgi_local.yaml",
        ]:
            config_path = CONFIGS_DIR / config_file
            config = OmegaConf.load(config_path)

            # Ensure config is a DictConfig (not ListConfig)
            if not OmegaConf.is_dict(config):
                continue

            # Cast to DictConfig for type safety
            config = OmegaConf.to_container(config, resolve=True)
            if not isinstance(config, dict):
                continue

            gen_config = config.get("generation", {})

            # Should have valid generation params
            assert "temperature" in gen_config
            assert "max_tokens" in gen_config
            assert "top_p" in gen_config

            # Validate they're in acceptable ranges
            gen_validated = GenerationConfig(**gen_config)
            assert 0.0 <= gen_validated.temperature <= 2.0
            assert gen_validated.max_tokens > 0
            assert 0.0 <= gen_validated.top_p <= 1.0


class TestOpenAICompatibleModelDirectParams:
    """Test model creation with direct parameters (without config files)."""

    def test_from_vllm_direct_params(self):
        """Test from_vllm with direct parameters."""
        model = OpenAICompatibleModel.from_vllm(
            base_url="http://localhost:8000/v1", model_name="test-model"
        )

        assert model.model_name == "test-model"
        assert model.base_url == "http://localhost:8000/v1/"

    def test_from_llamacpp_direct_params(self):
        """Test from_llamacpp with direct parameters."""
        model = OpenAICompatibleModel.from_llamacpp(
            base_url="http://localhost:8080/v1", model_name="test-model.gguf"
        )

        assert model.model_name == "test-model.gguf"
        assert model.base_url == "http://localhost:8080/v1/"

    def test_from_tgi_direct_params(self):
        """Test from_tgi with direct parameters."""
        model = OpenAICompatibleModel.from_tgi(
            base_url="http://localhost:3000/v1", model_name="test/model"
        )

        assert model.model_name == "test/model"
        assert model.base_url == "http://localhost:3000/v1/"

    def test_from_llamacpp_default_model_name(self):
        """Test that from_llamacpp uses default model name when not provided."""
        model = OpenAICompatibleModel.from_llamacpp(base_url="http://localhost:8080/v1")

        assert model.model_name == "llama"

    def test_from_custom_with_api_key(self):
        """Test from_custom with API key."""
        model = OpenAICompatibleModel.from_custom(
            base_url="https://api.example.com/v1",
            model_name="custom-model",
            api_key="secret-key",
        )

        assert model.model_name == "custom-model"


class TestLLMModelConfigValidation:
    """Test LLMModelConfig datatype validation."""

    def test_rejects_empty_model_name(self):
        """Test that empty model_name is rejected."""
        with pytest.raises(ValidationError):
            LLMModelConfig(
                provider=LLMProvider.VLLM,
                model_name="",
                base_url="http://localhost:8000/v1",
            )

    def test_rejects_whitespace_model_name(self):
        """Test that whitespace-only model_name is rejected."""
        with pytest.raises(ValidationError):
            LLMModelConfig(
                provider=LLMProvider.VLLM,
                model_name="   ",
                base_url="http://localhost:8000/v1",
            )

    def test_rejects_empty_base_url(self):
        """Test that empty base_url is rejected."""
        with pytest.raises(ValidationError):
            LLMModelConfig(provider=LLMProvider.VLLM, model_name="test", base_url="")

    def test_validates_timeout_positive(self):
        """Test that timeout must be positive."""
        with pytest.raises(ValidationError):
            LLMModelConfig(
                provider=LLMProvider.VLLM,
                model_name="test",
                base_url="http://localhost:8000/v1",
                timeout=0,
            )

        with pytest.raises(ValidationError):
            LLMModelConfig(
                provider=LLMProvider.VLLM,
                model_name="test",
                base_url="http://localhost:8000/v1",
                timeout=-10,
            )

    def test_validates_timeout_max(self):
        """Test that timeout has maximum limit."""
        with pytest.raises(ValidationError):
            LLMModelConfig(
                provider=LLMProvider.VLLM,
                model_name="test",
                base_url="http://localhost:8000/v1",
                timeout=700,
            )

    def test_validates_max_retries_range(self):
        """Test that max_retries is within valid range."""
        config = LLMModelConfig(
            provider=LLMProvider.VLLM,
            model_name="test",
            base_url="http://localhost:8000/v1",
            max_retries=5,
        )
        assert config.max_retries == 5

        with pytest.raises(ValidationError):
            LLMModelConfig(
                provider=LLMProvider.VLLM,
                model_name="test",
                base_url="http://localhost:8000/v1",
                max_retries=11,
            )

        with pytest.raises(ValidationError):
            LLMModelConfig(
                provider=LLMProvider.VLLM,
                model_name="test",
                base_url="http://localhost:8000/v1",
                max_retries=-1,
            )

    def test_strips_whitespace_from_model_name(self):
        """Test that whitespace is stripped from model_name."""
        config = LLMModelConfig(
            provider=LLMProvider.VLLM,
            model_name="  test-model  ",
            base_url="http://localhost:8000/v1",
        )

        assert config.model_name == "test-model"

    def test_strips_whitespace_from_base_url(self):
        """Test that whitespace is stripped from base_url."""
        config = LLMModelConfig(
            provider=LLMProvider.VLLM,
            model_name="test",
            base_url="  http://localhost:8000/v1  ",
        )

        assert config.base_url == "http://localhost:8000/v1"


class TestGenerationConfigValidation:
    """Test GenerationConfig datatype validation."""

    def test_validates_temperature_range(self):
        """Test that temperature is constrained to valid range."""
        config = GenerationConfig(temperature=0.7)
        assert config.temperature == 0.7

        GenerationConfig(temperature=0.0)
        GenerationConfig(temperature=2.0)

        with pytest.raises(ValidationError):
            GenerationConfig(temperature=2.1)

        with pytest.raises(ValidationError):
            GenerationConfig(temperature=-0.1)

    def test_validates_max_tokens(self):
        """Test that max_tokens is positive."""
        config = GenerationConfig(max_tokens=512)
        assert config.max_tokens == 512

        with pytest.raises(ValidationError):
            GenerationConfig(max_tokens=0)

        with pytest.raises(ValidationError):
            GenerationConfig(max_tokens=-100)

        with pytest.raises(ValidationError):
            GenerationConfig(max_tokens=40000)

    def test_validates_top_p_range(self):
        """Test that top_p is between 0 and 1."""
        config = GenerationConfig(top_p=0.9)
        assert config.top_p == 0.9

        GenerationConfig(top_p=0.0)
        GenerationConfig(top_p=1.0)

        with pytest.raises(ValidationError):
            GenerationConfig(top_p=1.1)

        with pytest.raises(ValidationError):
            GenerationConfig(top_p=-0.1)

    def test_validates_penalties(self):
        """Test that frequency and presence penalties are in valid range."""
        config = GenerationConfig(frequency_penalty=0.5, presence_penalty=0.5)
        assert config.frequency_penalty == 0.5
        assert config.presence_penalty == 0.5

        GenerationConfig(frequency_penalty=-2.0, presence_penalty=-2.0)
        GenerationConfig(frequency_penalty=2.0, presence_penalty=2.0)

        with pytest.raises(ValidationError):
            GenerationConfig(frequency_penalty=2.1)

        with pytest.raises(ValidationError):
            GenerationConfig(frequency_penalty=-2.1)

        with pytest.raises(ValidationError):
            GenerationConfig(presence_penalty=2.1)

        with pytest.raises(ValidationError):
            GenerationConfig(presence_penalty=-2.1)


class TestConfigurationPrecedence:
    """Test that configuration precedence works correctly."""

    def test_direct_params_override_config_model_name(self):
        """Test that direct model_name overrides config."""
        config_path = CONFIGS_DIR / "vllm_pydantic.yaml"
        config = OmegaConf.load(config_path)

        # Ensure config is a DictConfig (not ListConfig)
        assert OmegaConf.is_dict(config), "Config is not a dict config"
        # Cast to DictConfig for type safety
        dict_config: DictConfig = config  # type: ignore

        model = OpenAICompatibleModel.from_config(
            dict_config, model_name="override-model"
        )

        assert model.model_name == "override-model"

    def test_direct_params_override_config_base_url(self):
        """Test that direct base_url overrides config."""
        config_path = CONFIGS_DIR / "vllm_pydantic.yaml"
        config = OmegaConf.load(config_path)

        # Ensure config is a DictConfig (not ListConfig)
        assert OmegaConf.is_dict(config), "Config is not a dict config"
        # Cast to DictConfig for type safety
        dict_config: DictConfig = config  # type: ignore

        model = OpenAICompatibleModel.from_config(
            dict_config, base_url="http://override:9000/v1"
        )

        assert "override:9000" in model.base_url

    def test_env_vars_work_as_fallback(self):
        """Test that environment variables work as fallback."""
        with patch.dict(os.environ, {"LLM_BASE_URL": "http://env:7000/v1"}):
            config = OmegaConf.create({"provider": "vllm", "model_name": "test"})

            model = OpenAICompatibleModel.from_config(config)

            assert "env:7000" in model.base_url


class TestModelRequirements:
    """Test required parameters."""

    def test_from_vllm_requires_base_url(self):
        """Test that missing base_url raises error."""
        with pytest.raises((ValueError, TypeError)):
            OpenAICompatibleModel.from_vllm(model_name="test-model")

    def test_from_vllm_requires_model_name(self):
        """Test that missing model_name raises error."""
        with pytest.raises((ValueError, TypeError)):
            OpenAICompatibleModel.from_vllm(base_url="http://localhost:8000/v1")


class TestModelAliases:
    """Test model aliases."""

    def test_vllm_model_alias(self):
        """Test that VLLMModel is an alias for OpenAICompatibleModel."""
        assert VLLMModel is OpenAICompatibleModel

    def test_llamacpp_model_alias(self):
        """Test that LlamaCppModel is an alias for OpenAICompatibleModel."""
        assert LlamaCppModel is OpenAICompatibleModel


class TestModelProperties:
    """Test model properties and attributes."""

    def test_model_has_model_name_property(self):
        """Test that model exposes model_name property."""
        model = OpenAICompatibleModel.from_vllm(
            base_url="http://localhost:8000/v1", model_name="test-model"
        )

        assert hasattr(model, "model_name")
        assert model.model_name == "test-model"

    def test_model_has_base_url_property(self):
        """Test that model exposes base_url property."""
        model = OpenAICompatibleModel.from_vllm(
            base_url="http://localhost:8000/v1", model_name="test-model"
        )

        assert hasattr(model, "base_url")
        assert "localhost:8000" in model.base_url
