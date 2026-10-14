from knowledge.process.import_process.base import BaseNode
from knowledge.process.import_process.state import ImportGraphState
from knowledge.process.import_process.exceptions import ValidationError

from pathlib import Path



class EntryNode(BaseNode):
    """
    实体节点
    整个导入流程中位置第一位
    对上传的文件类型判断
    """
    name = 'entry'
    def process(self,state:ImportGraphState):
        """
        处理文件类型的检测
        Args:
            state: ImportGraphState

        Returns:ImportGraphState：该节点处理之后的状态

        """
        #1、获取导入的文件路径以及所在的目录
        self.log_step('step1', '[获取文件路径]')
        import_file_path= state.get('import_file_path')
        file_dir= state.get('file_dir')

        #2、校验文件路径及所在目录

        self.log_step('step2','[检测文件路径]')
        if not file_dir or not import_file_path:
            raise ValidationError('文件目录或文件不存在',self.name)

        #3、使用标准的path对象操作文件逻辑
        path = Path(import_file_path)

        #4、获取上传文件的后缀
        suffix= path.suffix.lower()

        #5、判断文件的后缀
        if suffix == '.pdf':
            state['is_pdf_read_enabled'] = True
            state['pdf_path']= import_file_path
        elif suffix == '.md':
            state['is_md_read_enabled'] = True
            state['md_path']= import_file_path
        else:
            self.logger.debug(f'文(件类型{suffix}不支持')
            raise ValidationError(f'文件类型{suffix}不支持')

        #6、获取文件的标题名
        file_title = path.stem
        state['file_title']= file_title

        #7、返回state
        return state



