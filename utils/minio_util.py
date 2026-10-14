import os
from minio import Minio
import logging
from dotenv import load_dotenv
load_dotenv()

# MinIO 配置（支持环境变量覆盖）
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME")


def get_minio_client(minio_access_key, minio_secret_key):
    # knowledge/tools/minio_utils.py



    # 初始化 MinIO 客户端
    try:
        minio_client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=False  # 本地开发不使用 HTTPS
        )
        # 确保 Bucket 存在
        bucket_name= os.getenv("MINIO_BUCKET_NAME")
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)
            logging.info(f"桶不存在{bucket_name}")
    except Exception as e:
        print(f"MinIO 创建 failed: {bucket_name}")
        minio_client = None

    def get_minio_client():
        """获取 MinIO 客户端单例"""
        return minio_client