from langchain_openai import ChatOpenAI
import os,logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
from dotenv import load_dotenv
load_dotenv()

cache_llm_client = {}

def get_llm_client(model_name:str=None,temperature:float=0,response_format:bool=False) -> ChatOpenAI:
    """

    Returns:llm客户端对象

    """
    model_name = model_name or os.getenv("MODEL_ITEM")
    api_key = os.getenv("API_KEY")
    api_base = os.getenv("API_BASE")

    cache_key= (model_name,response_format)
    if cache_key in cache_llm_client:
        return cache_llm_client[cache_key]

    model_kwargs= {}
    if response_format:
        model_kwargs["response_format"] = {"type":"json_object"}


    try:
        # 定义模型实例
        client = ChatOpenAI(
            model_name=model_name,
            openai_api_key=api_key,
            openai_api_base=api_base,
            temperature=temperature,
            extra_body={"enable_thinking": False},
            model_kwargs={}
        )

        cache_llm_client[cache_key] = client
        return client
    except Exception as e:
        logger.error(f'llm客户端创建失败')
        return None




