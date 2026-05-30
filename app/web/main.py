"""
Streamlit 前端界面（v0.2）

支持：
- 文件上传（multipart/form-data）
- 多轮对话历史
- 查看引用来源
- 删除已上传文档

启动命令：
    streamlit run app/web/main.py
"""

import streamlit as st
import requests
import json
from typing import List

API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="企业智能知识助手",
    page_icon="🤖",
    layout="wide"
)


def init_session_state():
    """初始化会话状态"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        import uuid
        st.session_state.session_id = str(uuid.uuid4())


def load_history():
    """从后端加载对话历史"""
    try:
        resp = requests.get(
            f"{API_BASE_URL}/api/v1/chat/history",
            params={"session_id": st.session_state.session_id},
            timeout=5
        )
        if resp.status_code == 200:
            data = resp.json()
            old_messages = st.session_state.messages.copy()
            new_messages = []
            for msg in data.get("messages", []):
                role = msg.get("role")
                content = msg.get("content")
                if not role or not content:
                    continue
                new_messages.append({
                    "role": role,
                    "content": content,
                    "sources": msg.get("sources", [])
                })
            st.session_state.messages = new_messages
    except Exception as e:
        st.sidebar.warning(f"加载历史失败: {e}")


def call_chat_api(query: str, enable_agent: bool = False) -> dict:
    """调用后端问答 API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/chat",
            json={
                "query": query,
                "session_id": st.session_state.session_id,
                "top_k": 15,
                "enable_agent": enable_agent
            },
            timeout=300
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ 无法连接到后端服务。请先启动 API：\n```\nuvicorn app.api.main:app --port 8000\n```")
        return None
    except Exception as e:
        st.error(f"❌ 请求失败: {str(e)}")
        return None


def upload_file(file) -> dict:
    """上传文件到后端"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/ingest",
            files={"file": (file.name, file.getvalue(), file.type)},
            timeout=180
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"上传失败: {e}")
        return None


def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        st.title("⚙️ 设置")

        st.session_state.enable_agent = st.toggle(
            "启用 Agent 模式",
            value=True,
            help="Agent 模式支持多步推理、对比分析等复杂任务"
        )

        # 加载历史
        if st.button("🔄 加载历史对话"):
            load_history()
            st.rerun()

        if st.button("🗑️ 清空当前对话"):
            st.session_state.messages = []
            try:
                requests.delete(
                    f"{API_BASE_URL}/api/v1/chat/history",
                    params={"session_id": st.session_state.session_id}
                )
            except:
                pass
            st.rerun()

        st.divider()

        # 知识库统计
        st.subheader("📊 知识库状态")
        if st.button("刷新统计"):
            try:
                resp = requests.get(f"{API_BASE_URL}/api/v1/stats", timeout=5)
                if resp.status_code == 200:
                    stats = resp.json()
                    st.metric("向量库文档数", stats.get("total_documents_in_store", 0))
                    st.metric("已上传文件数", stats.get("total_files_uploaded", 0))
            except Exception as e:
                st.warning(f"服务未启动: {e}")

        st.divider()

        # 已上传文档列表
        st.subheader("📁 已上传文档")
        try:
            resp = requests.get(f"{API_BASE_URL}/api/v1/documents", timeout=5)
            if resp.status_code == 200:
                docs = resp.json().get("documents", [])
                for doc in docs:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.caption(f"{doc['filename']} ({doc['chunk_count']} chunks)")
                    with col2:
                        if st.button("🗑️", key=f"del_{doc['doc_id']}"):
                            try:
                                requests.delete(
                                    f"{API_BASE_URL}/api/v1/documents/{doc['doc_id']}",
                                    timeout=10
                                )
                                st.rerun()
                            except Exception as e:
                                st.error(f"删除失败: {e}")
        except:
            st.caption("暂无文档")

        st.divider()

        # 文件上传（支持批量）
        st.subheader("⬆️ 上传新文档")
        uploaded_files = st.file_uploader(
            "选择文件（可多选）",
            type=["pdf", "docx", "pptx", "txt", "md", "html"],
            accept_multiple_files=True,
            help="支持 PDF、Word、PPT、Markdown 等格式，可一次选择多个文件批量上传"
        )

        # 显示已选择的文件列表
        if uploaded_files:
            st.caption(f"已选择 {len(uploaded_files)} 个文件：")
            for f in uploaded_files:
                size_mb = len(f.getvalue()) / 1024 / 1024
                st.caption(f"  • {f.name} ({size_mb:.1f} MB)")

        if uploaded_files and st.button("🚀 开始批量摄取"):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, file in enumerate(uploaded_files):
                progress = (i) / len(uploaded_files)
                progress_bar.progress(progress)
                status_text.text(f"[{i+1}/{len(uploaded_files)}] 正在处理: {file.name}...")

                result = upload_file(file)
                if result:
                    results.append(result)
                else:
                    results.append({"filename": file.name, "error": "上传失败"})

            progress_bar.progress(1.0)
            status_text.empty()

            # 显示汇总结果
            success_count = sum(1 for r in results if "error" not in r)
            update_count = sum(1 for r in results if r.get("is_update") and "error" not in r)
            new_count = success_count - update_count

            st.success(f"📊 批量摄取完成！成功 {success_count}/{len(uploaded_files)} 个")
            if new_count:
                st.caption(f"  • 新增: {new_count} 个")
            if update_count:
                st.caption(f"  • 覆盖: {update_count} 个")

            # 显示每个文件的详细结果
            with st.expander("📋 详细结果"):
                for r in results:
                    if "error" in r:
                        st.error(f"❌ {r['filename']}: {r['error']}")
                    else:
                        action = "覆盖" if r.get("is_update") else "新增"
                        st.caption(f"✅ [{action}] {r['filename']} → {r['chunks_ingested']} chunks")

            # 自动刷新文档列表
            st.rerun()

        st.divider()
        st.caption("企业智能知识助手 v0.2.0")


def render_chat_interface():
    """渲染聊天界面"""
    st.title("🤖 企业智能知识助手")
    st.caption("基于 RAG + Agent 架构的私有知识库问答系统")

    # 显示对话历史
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message and message["sources"]:
                with st.expander("📚 查看来源"):
                    for i, source in enumerate(message["sources"][:5], 1):
                        meta = source.get("metadata", {})
                        st.markdown(f"**[{i}]** {source.get('content', '')[:200]}...")
                        st.caption(
                            f"来源: {meta.get('source_file', 'N/A')} | "
                            f"页码: {meta.get('page_number', 'N/A')} | "
                            f"相关度: {source.get('score', 0):.3f}"
                        )

    # 用户输入
    if prompt := st.chat_input("请输入您的问题..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                result = call_chat_api(
                    prompt,
                    enable_agent=st.session_state.get("enable_agent", False)
                )

            if result:
                answer = result.get("answer", "无回答")
                sources = result.get("sources", [])
                query_time = result.get("query_time_ms")

                st.markdown(answer)

                if sources:
                    with st.expander("📚 查看来源"):
                        for i, source in enumerate(sources[:5], 1):
                            meta = source.get("metadata", {})
                            st.markdown(f"**[{i}]** {source.get('content', '')[:200]}...")
                            st.caption(
                                f"来源: {meta.get('source_file', 'N/A')} | "
                                f"页码: {meta.get('page_number', 'N/A')} | "
                                f"相关度: {float(source.get('score') or 0):.3f}"
                            )

                if query_time:
                    st.caption(f"⏱️ 响应时间: {query_time}ms")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })


def main():
    init_session_state()
    render_sidebar()
    render_chat_interface()


if __name__ == "__main__":
    main()
