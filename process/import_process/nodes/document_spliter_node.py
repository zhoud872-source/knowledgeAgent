import json
import os
import re
from pathlib import Path
from typing import Tuple, List, Dict, Any
from unittest import result

from knowledge.process.import_process.base import BaseNode,setup_logging,T
from knowledge.process.import_process.state import ImportGraphState
from knowledge.process.import_process.exceptions import ValidationError
from knowledge.process.import_process.config import get_config
from langchain_text_splitters import RecursiiveCharacterTextSplitter
from knowledge.utils.markdown_util import MarkdownTableLinearizer


class DocumentSpliterNode(BaseNode):
    def process(self,state:ImportGraphState)->ImportGraphState:
        # 加载 打散（嵌入模型准、注入元数据、多路召回、成本低） 组合
        # 获取md内容
        md_content,file_title,max_content_length,min_content_length = self._get_input(state)

        #1、根据标题切割
        sections,has_title =self._split_by_headings(md_content,file_title)

        #2、处理
        #2.1、section内容过长 继续进行二次切割    #3、内容过短，看能不能合并（如果不能合并 反之合并）
        final_chunks = self._split_and_merge(sections,max_content_length,min_content_length)


        #4、组装state:chunks
        chunks = self._assemble_chunk(final_chunks)

        # 更新state:chunks
        state["chunks"]=chunks

        # Step 6: 日志统计
        self._log_summary(md_content, chunks, max_content_length)

        # Step 7: 备份
        state["chunks"] = chunks
        self._backup_chunks(state, chunks)

        return state

        # ------------------------------------------------------------------ #
        #                       日志 & 备份                                    #
        # ------------------------------------------------------------------ #

    def _log_summary(self, raw_content: str, chunks: List[dict], max_length: int):
        self.log_step("step_6", "输出统计")

        lines_count = raw_content.count("\n") + 1
        self.logger.info(f"原文档行数: {lines_count}")
        self.logger.info(f"最终切分章节数: {len(chunks)}")
        self.logger.info(f"最大切片长度: {max_length}")

        if chunks:
            self.logger.info("章节预览:")
            for i, sec in enumerate(chunks[:5]):
                title = sec.get("title", "")[:50]
                self.logger.info(f"  {i + 1}. {title}...")
            if len(chunks) > 5:
                self.logger.info(f"  ... 还有 {len(chunks) - 5} 个章节")

    def _backup_chunks(self, state: ImportGraphState, sections: List[dict]):
        self.log_step("step_7", "备份切片")

        # 优先使用 file_dir，兼容 local_dir (避免因为字段命名引发写入失败)
        local_dir = state.get("file_dir", state.get("local_dir", ""))
        if not local_dir:
            self.logger.debug("未设置 file_dir/local_dir，跳过备份")
            return

        try:
            os.makedirs(local_dir, exist_ok=True)
            output_path = os.path.join(local_dir, "chunks.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(sections, f, ensure_ascii=False, indent=2)
            self.logger.info(f"已备份到: {output_path}")
        except Exception as e:
            self.logger.warning(f"备份失败: {e}")



    def _get_input(self, state:ImportGraphState)->Tuple[str,str,int]:
        config = get_config()
        #1、获取md_content
        md_content = state.get("md_content")

        #2、同意换行符
        if md_content:
            md_content.replace("\r\n","\n").replace("\r","\n")

        #3、文件标题
        file_title = state.get("file_title")

        return md_content,file_title,config.max_content_length

    def _split_by_headings(self, md_content:str, file_title:str)->Tuple[List[dict],bool]:
        """
        根据md的标题（1-6）切分
        Args:
            md_content: md内容
            file_title: 文档名字

        Returns:
            Tuple[List[dict],bool]
            section 用List装section
            bool md文档是否有标题

        """
        #1、定义变量
        in_fence = False
        has_title = False
        body_lines = []
        sections = []
        current_level = 0
        current_title= ""
        hierarchy = [""]*7 #0号位置不用

        # 定义正则表达式（标题）
        heading_re = re.compile(r"\s*(#{1,6})\s+(.+)")

        #3、切分
        content_lines = md_content.split("\n")

        def _flush():
            """
            封装section对象
            Returns:

            """
            body = "\n".join(body_lines)
            if current_title or body:

                parent_title = ""
                for i in range(current_level - 1, 0, -1):
                    if hierarchy[i]:
                        parent_title = hierarchy[i]
                        break
                if not parent_title:
                    parent_title =current_title if current_title else file_title

                return sections.append({
                    "title": current_title,
                    "body": body,
                    "file_title": file_title,
                    "parent_title": parent_title,
                })




        for content_line in content_lines:
            #3.1 判断 是否存在代码注释
            if content_line.startswith("'''") or content_line.strip().startwith("~~~"):
                in_fence = not in_fence

            match = heading_re.match(content_line) if not in_fence else None

            if match:
                has_title = True
                # 当前行是标题
                _flush()
                level= len(match.group(1))  #当前标题的级别
                current_level= level    #当前标题的级别

                current_title= content_line
                hierarchy[level]=current_title #当前标题的名字

                # 存储当前标题
                for i in range(level+1,7):
                    hierarchy[i]= ""

                body_lines = []
            else:
                body_lines.append(content_line)
        _flush()
        return sections,has_title

    def _split_and_merge(self, sections, max_content_length, min_content_length):
        """

        Args:
            sections: 根据以及标题切分后的块
            max_content_length: 每一个section内容最大长度
            min_content_length: 每一个section内容长度小则进行合并

        Returns:
            List[section]

        """
        self.log_step("step1","切分内容")

        # 切分
        current_sections = []
        for section in sections:
            current_sections.extend(self.split_long_section(section,max_content_length))


        # 合并(切割剩下的小块和原本的)
        final_sections = self.merge_short_section(current_sections,min_content_length)

        # 返回
        return final_sections

    def split_long_section(self,section:Dict[str,Any],max_content_length:int):
        """
        达到长度才切
        Args:
            section:
            max_content_length:

        Returns:

        """
        self.log_step("step3","长内容切分")
        #1、获取section对象的属性
        title = section.get("title")
        body = section.get("body")
        file_title = section.get("file_title")
        parent_title = section.get("parent_title")

        # 处理表格
        if "<table>" in body:
            self.logger.info("检测到表格")
            body= MarkdownTableLinearizer.process(body)


        MAX_title_length = 50
        if len(title) > MAX_title_length:
            self.logger.warning(f"文件{file_title}对应的{title}过长")
            title = title[:MAX_title_length]

        # 2、拼接title的前缀
        title_prefix = f"{title}\n\n"

        # 3、计算总长度（len(title_prefix)+len(body)）
        total_length = len(title_prefix) + len(body)

        # 4、判断
        if total_length <= max_content_length:
            return [section]

        #5、计算body可用长度
        body_length = max_content_length - len(title_prefix)
        if body_length <= 0:
            return [section]


        #6、切分

        text_splitter = RecursiiveCharacterTextSplitter(chunk_size=body_length,
                                                        chunk_overlap=0,
                                                        separators=["/n/n","/n","。","！",'？',"：",""],
                                                        keep_separator=False
                                                        )
        texts= text_splitter.split_text(body)

        if len(texts) <= 1:
            return [section]

        sub_section = []
        for index,text in enumerate(texts):
            sub_section.append({
                "title":title+f"{index+1}",
                "body":text,
                "file_title":file_title,
                "parent_title":parent_title,
                "part":f"{index+1}"
            })
        return sub_section

    def merge_short_section(self, current_sections, min_content_length):
        """
        贪心累加算法

        Args:
            current_sections:
            min_content_length:

        Returns:

        """
        current_section= current_sections[0]
        final_sections = []

        for next_section in current_sections[1:]:
           # 同源
           same_parent = (current_section["parent_title"] == next_section["parent_title"])

           if same_parent and len(current_section.get("body")) < min_content_length :
               # body merge
                current_section["body"] = (
                    current_section.get("body").rstrip()+ "\n\n" +next_section.get("body").lstrip()
                )
               # 更新current的title
                current_section["title"] = current_section["parent_title"]

                if 'part' not in current_section and 'part' in next_section:
                    current_section["part"] = 0

           else:
               final_sections.append(current_section)
               current_section = next_section

        final_sections.append(current_section)
               # part 更新
        part_counter={}
        for final_section in final_sections:
            if "part" in final_section:

                parent_title = final_section.get("parent_title")
                part_counter[parent_title] = part_counter.get(parent_title,0)+1

                new_part = part_counter[parent_title]

                final_section['part'] = new_part

                final_section["title"] = final_section["title"]+f"-{new_part}"

            result.append(final_section)


        return result

    def _assemble_chunk(self, final_chunks:List[str,Any])->List[Dict[str,Any]]:
        """
        最终组合chunk
        Args:
            final_chunks:

        Returns:

        """
        chunks = []
        for chunk in final_chunks:

            title = chunk.get("title")
            file_title = chunk.get("file_title")
            parent_title = chunk.get("parent_title")
            body = chunk.get("body")
            content= f"{title}\n\n{body}"

            assemble_chunk = {
                "title": title,
                "file_title": file_title,
                "parent_title": parent_title,
                "content": content,
            }

            # 判断part是否存在
            if "part" in chunk:
               assemble_chunk["part"] = chunk.get("part")
               chunks.append(assemble_chunk)

        return chunks








