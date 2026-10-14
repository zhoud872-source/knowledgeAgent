import base64
import logging
import time
from collections import deque
from pathlib import Path
from typing import Tuple, List, Deque
import os
import re
from openai import OpenAI

from knowledge.utils.minio_util import get_minio_client
from knowledge.process.import_process.base import BaseNode,setup_logging
from knowledge.process.import_process.state import ImportGraphState
from knowledge.process.import_process.exceptions import ValidationError, FileProcessingError, PdfConversionError, \
    ImageProcessingError
from knowledge.process.import_process.config import get_config


class MarkDownImageNode(BaseNode):
    """
    处理md节点内图片
    """
    name = "md_img_node"

    def process(self,state:ImportGraphState) -> ImportGraphState:
        """

        Args:
            state: 上一个节点处理之后state的最新状态

        Returns:当前节点处理之后的最新状态（md_contenr=process_md）

        """
        config = get_config()
        #1、处理文件路径(1.1md内容 1.2md的path 1.3图片目录)
        md_content, md_path_obj, img_dir= self._get_img_md_content(state)
        if not img_dir.exists():
            # 图片不用处理，直接更新md_content
            self.logger.warning(f"文件{md_path_obj.name} pdf中暂无图片")
            state["md_content"]=md_content
            return state


        #2、扫描处理并过滤有效图片（）
        target_images_context= self._scan_images_and_context(img_dir,md_content,config)

        #3、为图片生成图片描述（摘要）
        images_summaries = self._extract_img_summary(md_path_obj.stem,target_images_context,config)

        #4、图片上传到minio---->remote_url
        #5、回写（图片描述和图片地址）写到md_content  以及摘要

        new_md_content = self._upload_img_and_update_new_md(md_path_obj.stem,md_content,images_summaries,target_images_context,config)

        self._backup_new_md_file(md_path_obj, new_md_content)


        #6、返回state
        state["md_content"]=new_md_content

        return state

    # 更新后的内容备份（调试）


    def _backup_new_md_file(
            self,
            md_path_obj: Path,
            new_md_content: str
    ) -> str:

        self.log_step("step_5", "备份新文件")

        original_path = Path(md_path_obj)
        new_file_path = original_path.with_name(
            f"{md_path_obj.stem}_new{md_path_obj.suffix}"
        )

        try:
            with open(new_file_path, "w", encoding="utf-8") as f:
                f.write(new_md_content)
            self.logger.info(f"处理后的文件已备份至: {new_file_path}")
        except IOError as e:
            self.logger.error(f"写入新文件失败 {new_file_path}: {e}")
            raise ImageProcessingError(f"文件写入失败: {e}", node_name=self.name)

        return str(new_file_path)

    def _get_img_md_content(self,state:ImportGraphState)-> Tuple[str,Path,Path]:
        """

        Args:
            state: 上一个节点处理之后的最新state

        Returns:
            md content：md的内容
            md_path: md的路径
            image_dir:图片目录

        """
        self.log_step("step1","读取md内容以及构建图片目录")
        #1、获取md_path
        md_path= state.get("md_path","")

        #2、判断是否有内容
        if not md_path:
            raise ValidationError("md文件不存在",self.name)

        #3、标准化数据
        md_path_obj = Path(md_path)

        #4、判断路径是否有效
        if not md_path_obj.exists():
            raise FileProcessingError("md文件路径无效",self.name)

        #5、读取文件内容
        with open(md_path_obj,"r",encoding="utf-8") as f:
            md_content = f.read()

        #6、构建图片目录
        img_dir = md_path_obj.parent / "images"

        #7、返回
        return md_content,md_path_obj,img_dir

    def _scan_images_and_context(self, img_dir:Path, md_content:str,config) -> List[Tuple[str,str,Tuple[str,str,str]]]:
        """
        扫描并处理图片,上下文
        返回所有有效图片的丰富信息（图片img_name,image_path,图片上下文）
        图片上下文的作用：图片的上文和下文 对上文、下文max token确定
        找图片位置，1个最近上文标题（定位标题的位置与内容）标题内容从上往下取，根据开始和结束索引定位内容，利用段落和最大字符数选择内容
        Args:

            img_dir: 图片目录
            md_content：md内容
            config:配置信息

        Returns:
            List["图片名字"，“图片地址”]

        """
        self.log_step("step2",f"扫描文件目录{img_dir}")
        config =get_config()
        #1、遍历文件目录
        target_images_context = []

        for img_name in os.listdir(img_dir):
            file_ext =os.path.splitext(img_name)[1] #获取文件后缀
            if file_ext not in config.image_extensions:
                continue

            # 1.2 否则构建image_path
            img_path = str (img_dir / img_name)

            #1.3 构建图片的上下文

            img_context= self._find_img_context_with_limit(md_content,img_name,config.img_content_length)
            if not img_context:
                self.logger.warning("md文件中无有效图片")
                continue

            # 1.4提取到当前图片的唯一上下文内容
            primary_img_context = img_context[0]

            #1.5 存储到列表中

            target_images_context.append((img_name ,img_path,primary_img_context))

            self.logger.info(f"找到{len(target_images_context)}有效的图片")

        return target_images_context

    def _find_img_context_with_limit(self,md_content:str,img_name:str,max_chars=200)-> List[Tuple[str,str,str]]:
         """
         从md文档中提取图片上下文信息
         Args:
             md_content: 要操作的信息
             img_name: 要定位的图片
             max_chars:最大字符数

         Returns:
             List[Tuple[""图片标题"",“图片上文”,“图片下文”]]

         """

         #1、使用正则查找md图片的位置(！[](image/aaa.jpg"提示"))
         re_pattern= re.compile(r"!\[.*?\]\(.*?" + re.escape(img_name) + r".*?\)")

         #2、从md中定位图片位置（按行切分，看是否满足图片）
         md_lines = md_content.split("\n")
         imgs_context= []
         for line_idx,line in enumerate(md_lines):
             if not re_pattern.search(line):
                 continue #继续下一行

             #b、找到了（索引）1.1、定位这张图片最近的标题 1.2、提取上文内容 1.3、提取下文内容

             #1.1、找标题 r"^#{1,6}\s+", lines[i]   找上文
             head_title = ""
             head_index = -1
             for line_number in range(line_idx - 1,-1,-1):
                 if re.match(r"^#{1,6}\s+", md_lines[line_number]):
                     head_title = md_lines[line_number]
                     head_index = line_number
                     break

             #1.2 截取上文的内容
             pre_content_start_index = head_index +1
             pre_content_end_index = line_idx
             pre_content = md_lines[pre_content_start_index:pre_content_end_index]

             img_pre_context= self._extract_img_context_with_limit(pre_content,max_chars,direction="front")

             # 找图片的下文内容(下文标题不要)
             md_line= len(md_lines)
             section_index = md_line
             for i in range(line_idx+1,md_line):
                 if re.match(r"^#{1,6}\s+", md_lines[i]):
                     section_index = i
                     break

             # 定义要截取的下文索引
             post_content_start_index = line_idx+1
             post_content_end_index = section_index
             post_content = md_lines[post_content_start_index:post_content_end_index]
             img_post_context = self._extract_img_context_with_limit(post_content,max_chars,direction="end")

             imgs_context.append((head_title,img_pre_context,post_content)) #实际上只有一个3元组对象

         return imgs_context

    def _extract_img_context_with_limit(self, extract_content:list, max_chars:int, direction:str)-> str:
       """
       提取图片上下文、
       direction:front 自下而上
       direction：end 自上而下
       如何从给定的段落找内容、md段落用\n分割，每一行有两个空格
       Args:
           extract_content: 提取到的内容
           max_chars: 最大字符数
           direction: 方向

       Returns:
           str:上下文

       """
       #1、切分内容的段落
       lines= extract_content.split("\n")

       current_paragraph= []
       final_paragraph = [] #有可能多个段落
       for line in lines:
           clean_strip= line.strip()
           if not clean_strip:
               if current_paragraph:
                    final_paragraph.append("\n".join(current_paragraph))  #[a,b,c] a\nb\nc
                    current_paragraph = []
           else:
               if re.match(r"^!\[.*?\]\(.*?\)$", clean_strip):
                   if current_paragraph:
                       final_paragraph.append("\n".join(current_paragraph))  # [a,b,c] a\nb\nc
                       current_paragraph = []
                   continue
               current_paragraph.append(line)
        #最后一段且下面没有空行
       if current_paragraph:
           final_paragraph.append("\n".join(current_paragraph))

       if direction == "front":
           final_paragraph.reverse()


       # 3、判断final中收集到段落长度
       total = 0
       selected= []
       for para in final_paragraph:
           para_len = len(para)

           if total + para_len > max_chars and selected:
               break
           selected.append(para)
           total += para_len

       # 4、返回selected
       if direction == "front":
           selected.reverse()
       return "\n\n".join(selected)

    def _extract_img_summary(self, document_title:str,
                             target_images_context:List[Tuple[str,str,Tuple[str,str,str]]],
                             config):
        """
        为所有图片生成图片摘要  VLM模型
        Args:
            document_title: 文档名字
            target_images_context: 所有图片信息
            config: 配置信息

        Returns: Dict{图片名字：摘要}

        """
        self.log_step("step3","准备提取图片摘要")
        summarise= {}
        request_timestamps: Deque[float] = deque()

        #1、构建openai客户端
        try:
            client= OpenAI(
                api_key=config.OPENAI_API_KEY,
                base_url=config.OPENAI_API_BASE,
            )
        except Exception as e:
            logging.error(f"VLM客户端创建失败")
            return summarise



        #2、发送请求（提取摘要）
        for img_name,img_path,images_context in target_images_context:
            self._enforce_rate_limit(request_timestamps,config.requests_per_minute,60)

            summary= self._get_img_summary(config,client,document_title, img_path,images_context)
            summarise[img_name] = summary


        #3、返回映射表
        logging.info(f"生成{len(summarise)}图片摘要")
        return summarise

    def _enforce_rate_limit(
            self,
            request_timestamps: Deque[float],
            max_requests: int,
            window_seconds: int = 60
    ):
        """
        强制执行 API 请求速率限制。

        Args:
            request_timestamps (Deque[float]): 请求时间戳队列。
            max_requests (int): 窗口内最大请求数。
            window_seconds (int, optional): 时间窗口大小（秒）。
        """
        current_time = time.time()

        # 移除窗口外的时间戳
        while request_timestamps and current_time - request_timestamps[0] >= window_seconds:
            request_timestamps.popleft()

        # 达到上限则等待
        if len(request_timestamps) >= max_requests:
            sleep_duration = window_seconds - (current_time - request_timestamps[0])
            if sleep_duration > 0:
                self.logger.info(f"达到速率限制，暂停 {sleep_duration:.2f} 秒...")
                time.sleep(sleep_duration)

            current_time = time.time()
            while request_timestamps and current_time - request_timestamps[0] >= window_seconds:
                request_timestamps.popleft()

        request_timestamps.append(current_time)


    def _get_img_summary(self, config, client, document_title:str, img_path:str, images_context:Tuple[Tuple[str,str,str]])->str:

        #1、 解包image context构建上下文
        section_title,pre_context,post_context= images_context

        #2、判断上下文
        context_parts= []
        if section_title:
            context_parts.append(section_title)
        if pre_context:
            context_parts.append(pre_context)
        if post_context:
            context_parts.append(post_context)
        final_context= "\n".join(context_parts) if context_parts else "暂无可用上下文"

        #4、读取图片文件
        local_img_content =""
        try:
            with open(img_path,"rb") as f:
                content =f.read()
                local_img_content =base64.b64encode(content).decode("utf-8")
        except Exception as e:
            return "暂无图片"

        #5、发送请求
        try:
            response = client.chat.completions.create(
                model=config.MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"""任务：为Markdown文档中的图片生成一个简短的中文标题。
                                 背景信息：
                                 1. 所属文档标题："{document_title}"
                                 2. 图片上下文： {final_context}
                                 请结合图片视觉内容和上述上下文信息，用中文简要总结这张图片的内容，
                                 生成一个精准的中文标题（不要包含"图片"二字）。""",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{local_img_content}"
                                }
                            }
                        ]
                    }
                ],
            )
            summary = response.choices[0].message.content.strip()
            return summary
        except Exception as e:
            self.logger.warning(f"图片摘要生成失败 {img_path}: {e}")
            return "图片描述"

    def _upload_img_and_update_new_md(self,document_name, md_content, images_summaries, target_images_context,config):
       """
       上传图片到minio以及md中的图片的url和摘要
       Args:
           md_content:
           images_summaries:
           target_images_context:

       Returns:

       """
       self.log_step("step5","上传图片并且更新md的摘要和图片地址到minio")
       remote_urls= {}
       # 1、构建minio的客户端
       minio_client= get_minio_client()

       if minio_client is None:
           self.logger.warning(f"无minio客户端")


       #2、遍历图片信息列表
       for img_name,img_path,images_context in target_images_context:

           object_name = f"{document_name}_{img_name}"
           try:
               minio_client.fput_objimg(
                   config.minio_bucket,
                   object_name,
                   img_path,
               )

               # 手动拼接远程地址
               remote_url = config.get_minio_base_url() + "/" + config.minio_bucket + "/" + object_name
               self.logger.info(f"图片{img_name}上传minio成功")
               remote_urls[object_name] = remote_url

           except Exception as e:
               self.logger.warning(f"{img_name}上传到minio失败")
               remote_urls[img_name] = "https://minio_mock" + document_name+ "/" + img_name

       self.logger.info(f"成功上传{len(remote_urls)}图片到minio")

       #3、摘要和图片地址替换到md中
       new_md_content= md_content
       for img_name,images_summary in images_summaries.items():
           remote_url = remote_urls.get(img_name)
           if not remote_url:
               continue

           #替换url和摘要
           replace_pattern = re.compile(
               r"!\[(.*?)\]\((.*?" + re.escape(img_name) + r".*?)\)",
               re.IGNORECASE
           )
           new_md_content = replace_pattern.sub(f"![{images_summary}({remote_url})]",new_md_content)

       return new_md_content





















