import subprocess
from pathlib import Path
from typing import Tuple

from knowledge.process.import_process.base import BaseNode,setup_logging
from knowledge.process.import_process.state import ImportGraphState
from knowledge.process.import_process.exceptions import ValidationError,FileProcessingError,PdfConversionError


class PdfToMdNode(BaseNode):
    """
    pdf转换md节点
    """
    name = 'pdf_to_md_node'

    def process(self,state: ImportGraphState) -> ImportGraphState:
        """

         Args:
             self:
             state:

         Returns:

        """
        #1、对参数校验（路径）
        import_file_path,file_dir_path =self._validate_state_inputs_path(state)

         #2、利用MInerU工具解析pdf为md
        processed_code= self._execute_mineru(import_file_path,file_dir_path)
        if processed_code != 0:
            raise FileProcessingError("mieru解析pdf失败",self.name)

         #3、获取md path
        md_path= self._get_md_path(import_file_path,file_dir_path)


        #4、更新state字典的md_path
        state['md_path']= md_path


          #5、返回state
        return state

    def _validate_state_inputs_path(self,state: ImportGraphState)-> Tuple[Path,Path]:
        """

        Args:
            self:
            state: 该节点收到的状态

        Returns:

        """
        self.log_step("step1","对状态的路径输入参数校验")
        #获取输入pdf文件路径
        import_file_path= state.get('import_file_path','')

        #获取解析后的输出目录
        file_dir= state.get('file_dir','')

        #校验输出文件路径
        if not import_file_path:
            raise ValidationError('解析的文件不存在',self.name)

        #4、用path标准化
        import_file_path_obj= Path(import_file_path)

        #5、校验是否真实有效
        if not import_file_path_obj.exists():
            raise FileProcessingError("解析的文件路径不存在",self.name)

        #6、判断输出目录是否为空
        if not file_dir:
            #默认目录兜底
            file_dir= import_file_path_obj.parent

        #7、标准化输出目录
        file_dir_path_obj= Path(file_dir)
        self.logger.info(f'输入文件的路径：{import_file_path}')
        self.logger.info(f'输出文件的目录：{file_dir}')

        #8、返回输出文件以及输出目录的标准path
        return import_file_path_obj,file_dir_path_obj

    def _execute_mineru(self,import_file_path,file_dir_path):
        """

        Args:
            import_file_path:解析文件路径
            file_dir_path: 解析后文件目录

        Returns:
            mineru -p <input_path> -o <output_path> --source local

        """
        #执行命令行
        self.log_step("step2","执行mineru解析pdf")

        #1、构建命令行
        cmd =["mineru",
              "-p",
              str(file_dir_path),
              "-o",
              str(import_file_path),
              "--source",
              "local"
        ]

        import time
        process_start_time = time.time()


        #2、执行命令行（子进程处理）自动读取主进程环境变量
        proc= subprocess.Popen(args=cmd,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT,
                         text=True,
                         encoding="utf-8",
                         errors="replace", #?
                         bufsize=1  #换行符，按行缓冲区，只要一行满了就换
                         )

        #3、获取日志信息
        for outlog in proc.stdout:
            self.logger.info(f"执行mineru产生的日志：{outlog}")

        #4、等待子进程做完(主进程等待子进程做完) 如果状态码等到子进程做完得到的状态码是0 代表子进程成功执行完了，反之，没执行成功
        processed_code= proc.wait()
        process_end_time = time.time()

        if processed_code == 0:
            self.logger.info(f"mineru执行成功，pdf文件:{import_file_path.name} 耗时：{process_end_time-process_start_time:.2f}")
        else:
            self.logger.error("pdf文件解析失败")

        #5、返回状态码
        return processed_code


    def _get_md_path(self,import_file_path:Path,file_dir_path:Path):
        file_name= import_file_path.stem

        md_path = file_dir_path / file_name / "hybrid_auto"/ f"{file_name}.md"
        return str(md_path)






##############测试







