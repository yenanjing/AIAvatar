import os
import yaml
from pathlib import Path


def load_config(config_path: str = None) -> dict:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径，默认为项目根目录下的 config.yml
        
    Returns:
        配置字典
    """
    if config_path is None:
        # 获取项目根目录（src 的父目录）
        current_dir = Path(__file__).parent
        config_path = current_dir.parent / "config.yml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config or {}


# 全局配置对象
_config = None


def get_config() -> dict:
    """
    获取配置（单例模式）
    
    Returns:
        配置字典
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config


def get_llm_config() -> dict:
    """
    获取 LLM 配置
    
    Returns:
        LLM 配置字典
    """
    config = get_config()
    return config.get('LLM', {})


def get_tts_config() -> dict:
    """
    获取 TTS 配置
    
    Returns:
        TTS 配置字典
    """
    config = get_config()
    return config.get('TTS', {})


# 便捷函数
def get_llm_api_key() -> str:
    """获取 LLM API Key"""
    return get_llm_config().get('LLM_API_KEY', '')


def get_llm_base_url() -> str:
    """获取 LLM Base URL"""
    return get_llm_config().get('LLM_BASE_URL', '')


def get_doubao_appid() -> str:
    """获取豆包 TTS AppID"""
    return get_tts_config().get('DOUBAO_APPID', '')


def get_doubao_token() -> str:
    """获取豆包 TTS Token"""
    return get_tts_config().get('DOUBAO_TOKEN', '')

