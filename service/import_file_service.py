from dotenv import load_dotenv

load_dotenv()
import os.path
import uuid
from datetime import datetime
import shutil
from typing import Tuple
from fastapi import UploadFile, HTTPException
from knowledge.core.paths import get_local_base_dir
from knowledge.utils.minio_util import get_minio_client
from knowledge.service.task_service import TaskService
from knowledge.process.import_process.state import ImportGraphState
from knowledge.process.import_process.main_graph import kb_import__graph_app



class ImportFileService:
    """
    文件导入的业务类
    保存上传的文件到本地、minio、运行图谱的节点
    """

    def __init__(self,task_service:TaskService):
        self._task_service=task_service





    def get_data_dir(self)->str:
        return os.path.join(get_local_base_dir(), datatime.now().strftime('%Y%m%d'))

    def save_upload_file_to_local(self,file:UploadFile,file_dir:str)->str:
        """
        上传的文件保存到本地
        Args:
            file:
            file_dir:

        Returns:

        """
        os.makedirs(file_dir,exist_ok=True)

        #构建上传文件完整的path
        import_file_path = os.path.join(file_dir,file.filename)

        #写入
        with open(import_file_path,'wb') as f:
            shutil.copyfileobj(file.file, f)

        # 返回导入文件的path
        return import_file_path





    def save_upload_file_to_minio(self,import_file_path:str,file:UploadFile)->Tuple[str,str,str]:

        minio_client = get_minio_client()

        if not minio_client:
            raise HTTPException(status=500,detail='minio client not available')

        minio_object_name = f'origin_files/{datatime.now().strftime("%Y%m%d")}/{file.filename}'

        bucket_name = os.getenv('MINIO_BUCKET_NAME')

        try:
            minio_client.fput_object(bucket_name,minio_object_name,import_file_path)
        except Exception as e:
            raise ValueError(f"{file.filename}上传失败  原因:{e}")

        return minio_object_name


    def process_upload_file(self,file:UploadFile)->str:
        """

        Returns:标记当前文件上传节点初始状态（upload file）\保存到本地、minio、标记运行完毕
        返回task_id\file_dir\import_file_path
        """
        #1、构建时间日期
        data_dir = self.get_data_dir()

        #2、生成任务id
        task_id= str(uuid.uuid4())

        #3、构建最终归属
        file_dir= os.path.join(data_dir, task_id)
        self._task_service.mark_node_running(task_id,"upload_file")

        #4、将接受到的文件上传到本地
        import_file_path= self.save_upload_file_to_local(file,file_dir)

        #5、将本地磁盘的上传的文件同步到minio
        self.save_upload_file_to_minio(import_file_path,file)
        self._task_service.mark_node_done(task_id,"upload_file")

        #6、构建返回的数据结构
        return task_id,file_dir,import_file_path


    def run_import_graph(self,task_id:str,file_dir:str,import_file_path:str)->Tuple[str,str,str]:
        """
        运行导入graph的流程
        Args:
            task_id:
            file_dir:
            import_file_path:

        Returns:

        """
        #1、构建初始状态

        #2、运行graph，运行某个节点之前调用mark_node_running，运行节点结束调用mark_node_done

        try:
            # 1. 标记任务开始处理
            self._task_service.update_task_status(task_id, "processing")

            # 2. 构建 LangGraph 初始状态
            global_graph_init_status: ImportGraphState = {
                "task_id": task_id,
                "file_dir": file_dir,
                "import_file_path": import_file_path
            }

            # 3. 流式执行整个导入流水线
            for event in kb_import_graph_app.stream(global_graph_init_status):
                for key, value in event.items():
                    print(f"[{task_id}] Completed Node: {key}")

            # 4. 标记任务完成
            self._task_service.update_task_status(task_id, "completed")

        except Exception as e:
            self._task_service.update_task_status(task_id, "failed")
            print(f"[{task_id}] Error: {e}")