import os
import tempfile
import hashlib
import shutil
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import json
import numpy as np

# RAG 관련 imports
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory
from langchain_core.runnables import RunnableMap

from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings


import hashlib
import shutil

# ✅ 파일 해시 생성
def get_file_hash(uploaded_file):
    file_content = uploaded_file.read()
    uploaded_file.seek(0)
    return hashlib.md5(file_content).hexdigest()

# ✅ pysqlite3 패치
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_chroma import Chroma
os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_API_KEY']

# ✅ CSV 로딩 → 유저 단위로 문서 생성
@st.cache_resource
def load_csv_and_create_docs(file_path: str):
    df = pd.read_csv(file_path)

    if 'user_id' not in df.columns or 'SPLITTED' not in df.columns:
        st.error("해당하는 컬럼 없음")
        return []

    docs = []
    for idx, row in df.iterrows():
        content = str(row['SPLITTED'])  # 한 행의 SPLITTED 값
        metadata = {"source": f"row_{idx}"}  # 행 인덱스를 소스로 사용
        docs.append(Document(page_content=content, metadata=metadata))
    return docs

@st.cache_resource
def get_embedder():
    class STEmbedding(Embeddings):
        def __init__(self, model_name: str):
            # ko 전용 임베딩 모델
            self.model = SentenceTransformer(model_name)

        def embed_documents(self, texts):
            # 리스트 입력에 대해 배치 인코딩
            return self.model.encode(list(texts), normalize_embeddings=True).tolist()

        def embed_query(self, text):
            # 단일 쿼리 인코딩
            return self.model.encode(text, normalize_embeddings=True).tolist()

    return STEmbedding("all-MiniLM-L6-v2")  #dragonkue/snowflake-arctic-embed-l-v2.0-ko

# ✅ 벡터스토어 생성
@st.cache_resource
def create_vector_store(file_path: str):
    docs = load_csv_and_create_docs(file_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(docs)

    file_hash = os.path.splitext(os.path.basename(file_path))[0]
    persist_dir = f"./chroma_db_user/{file_hash}"
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    embeddings = get_embedder()  # ← 여기만 교체
    vectorstore = Chroma.from_documents(
        split_docs,
        embeddings,
        persist_directory=persist_dir
    )
    return vectorstore

# ✅ RAG 체인 초기화
@st.cache_resource
def initialize_components(file_path: str, selected_model: str):
    vectorstore = create_vector_store(file_path)
    retriever = vectorstore.as_retriever()

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "이전 대화 내용을 반영해 현재 질문을 독립형 질문으로 바꿔줘."),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ])

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "다음 문서 내용을 참고하여 질문에 무조건 한국어로 답변해줘. 문서와 유사한 내용이 없으면 무조건 '관련된 내용이 없습니다'라고 말해줘. 꼭 이모지 써줘! 참고 문서는 아래에 보여줄 거야.\n\n{context}"),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ])

    llm = ChatOpenAI(model=selected_model)
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain



def create_tree_data_from_csv(df):
    """
    CSV 데이터에서 트리 데이터 구조를 생성하는 함수
    """
    # summary_table 생성
    summary_table = df[df.total_cl != 99].groupby(['keywords','summary'], as_index=False, dropna=False).agg({'user_id': 'nunique'}).rename(columns={'user_id': 'cnt'})
    
    # 키워드별로 총 cnt 계산 (노드 크기 결정용)
    keyword_totals = summary_table.groupby('keywords')['cnt'].sum().to_dict()
    
    # 색상 팔레트 생성 (키워드 개수만큼)
    colors = ['#ef4444', '#10b981', '#8b5cf6', '#f59e0b', '#06b6d4', 
              '#ec4899', '#84cc16', '#f97316', '#6366f1', '#14b8a6',
              '#f43f5e', '#22c55e', '#a855f7', '#eab308', '#0ea5e9']
    
    unique_keywords = summary_table['keywords'].unique()
    keyword_colors = {keyword: colors[i % len(colors)] for i, keyword in enumerate(unique_keywords)}
    
    # 트리 데이터 구조 생성
    tree_data = {
        'id': 'root',
        'name': '주요 응답',
        'expanded': False,
        'children': []
    }
    
    # 키워드별로 브랜치 노드 생성
    for keyword in unique_keywords:
        if pd.isna(keyword):
            keyword_name = '키워드 없음'
            keyword_id = 'no_keyword'
        else:
            keyword_name = str(keyword)
            keyword_id = f"keyword_{keyword_name.replace(' ', '_')}"
        
        keyword_summaries = summary_table[summary_table['keywords'] == keyword]
        
        # 해당 키워드의 summary들을 children으로 생성
        children = []
        for _, row in keyword_summaries.iterrows():
            summary_name = str(row['summary']) if pd.notna(row['summary']) else '요약 없음'
            summary_id = f"summary_{len(children)}"
            
            children.append({
                'id': f"{keyword_id}_{summary_id}",
                'name': summary_name,
                'color': keyword_colors[keyword],
                'cnt': int(row['cnt']),
                'type': 'summary'
            })
        
        # 키워드 브랜치 노드 생성
        branch_node = {
            'id': keyword_id,
            'name': keyword_name,
            'color': keyword_colors[keyword],
            'expanded': False,
            'cnt': keyword_totals[keyword],
            'children': children,
            'type': 'keyword'
        }
        
        tree_data['children'].append(branch_node)
    
    # cnt 값을 기준으로 정렬 (큰 값부터)
    tree_data['children'].sort(key=lambda x: x['cnt'], reverse=True)
    
    return tree_data

def calculate_dynamic_height(tree_data):
    """
    트리 데이터를 기반으로 필요한 높이를 계산하는 함수
    """
    if not tree_data.get('children'):
        return 400  # 기본 높이
    
    keyword_count = len(tree_data['children'])
    max_summary_count = max([len(child.get('children', [])) for child in tree_data['children']], default=0)
    
    # 높이 계산 공식
    base_height = 200  # 기본 여백
    keyword_height = keyword_count * 70  # 키워드당 70px
    summary_height = max_summary_count * 30  # 최대 요약 개수 * 30px
    
    total_height = max(400, base_height + keyword_height + summary_height)
    return min(total_height+100, 5000)  # 최대 1200px로 제한

def create_hierarchical_mindmap_from_data(tree_data):
    """
    계층형 마인드맵을 생성하는 함수
    """
    
    # 최대/최소 cnt 값으로 노드 크기 정규화
    all_cnts = []
    def collect_cnts(node):
        if 'cnt' in node:
            all_cnts.append(node['cnt'])
        if 'children' in node:
            for child in node['children']:
                collect_cnts(child)
    
    collect_cnts(tree_data)
    max_cnt = max(all_cnts) if all_cnts else 1
    min_cnt = min(all_cnts) if all_cnts else 1

    # 동적 높이 계산 - 이 부분 추가!
    dynamic_height = calculate_dynamic_height(tree_data)

    # HTML/CSS/JavaScript 코드
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Hierarchical MindMap</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
                overflow: auto;
                min-height: {str(dynamic_height)}vh;
                padding: 20px;
            }}
            
            .mindmap-container {{
                position: relative;
                width: 100%;
                min-height: {str(dynamic_height-50)}px;
                background: white;
                border-radius: 15px;
                box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
                padding: 40px;
                overflow: visible;
                overflow-y: auto;
                overflow-x: hidden;
            }}
            
            .root-node {{
                position: absolute;
                left: 50px;
                top: 50%;
                transform: translateY(-50%);
                width: 140px;
                height: 70px;
                background: linear-gradient(135deg, #3b82f6, #1d4ed8);
                border-radius: 15px;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: bold;
                font-size: 16px;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: 0 5px 20px rgba(59, 130, 246, 0.3);
                z-index: 10;
            }}
            
            .root-node:hover {{
                transform: translateY(-50%) scale(1.05);
                box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4);
            }}
            
            .keyword-node {{
                position: absolute;
                color: white;
                font-weight: 600;
                font-size: 14px;
                cursor: pointer;
                transition: all 0.3s ease;
                border-radius: 10px;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                z-index: 5;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                text-align: center;
            }}
            
            .keyword-node:hover {{
                transform: scale(1.1);
                box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
                z-index: 8;
            }}
            
            .summary-node {{
                position: absolute;
                color: white;
                font-size: auto;
                word-wrap: break-word;
                text-align: center;
                transition: all 0.2s ease;
                border-radius: 8px;
                box-shadow: 0 3px 10px rgba(0, 0, 0, 0.15);
                z-index: 3;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                line-height: 1.4;
                padding: 8px;
                width: auto;
                min-width: fit-content;
                overflow: visible;
            }}
            
            .summary-node:hover {{
                transform: scale(1.05);
                z-index: 6;
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.25);
            }}
            
            .connection-line {{
                position: absolute;
                z-index: 1;
            }}
            
            .main-branch {{
                stroke: #64748b;
                stroke-width: 3;
                fill: none;
            }}
            
            .sub-branch {{
                stroke-width: 2;
                fill: none;
                opacity: 0.8;
            }}
            
            .title {{
                position: absolute;
                top: 10px;
                left: 20px;
                z-index: 20;
                color: #1e293b;
            }}
            
            .expand-indicator {{
                margin-left: 8px;
                font-size: 16px;
                font-weight: bold;
            }}
            
            .cnt-indicator {{
                font-size: 11px;
                background: rgba(255, 255, 255, 0.25);
                padding: 2px 8px;
                border-radius: 12px;
                margin-top: 4px;
                backdrop-filter: blur(10px);
            }}
            
            .tooltip {{
                position: absolute;
                background: rgba(30, 41, 59, 0.95);
                color: white;
                padding: 12px 16px;
                border-radius: 8px;
                font-size: 13px;
                pointer-events: none;
                z-index: 100;
                max-width: 250px;
                word-wrap: break-word;
                box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
                backdrop-filter: blur(10px);
            }}
            
            .mindmap-svg {{
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                z-index: 1;
                pointer-events: none;
            }}
        </style>
    </head>
    <body>
        
        <div class="mindmap-container" id="mindmapContainer">
            <svg class="mindmap-svg" id="connectionSvg"></svg>
            
            <div class="root-node" id="rootNode" onclick="toggleRoot()">
                주요 응답
            </div>
        </div>
        
        <div id="tooltip" class="tooltip" style="display: none;"></div>

        <script>
            // 트리 데이터 (Python에서 생성된 데이터)
            const treeData = {json.dumps(tree_data, ensure_ascii=False)};
            const maxCnt = {max_cnt};
            const minCnt = {min_cnt};

            function getNodeSize(cnt, type) {{
                const normalizedCnt = (cnt - minCnt) / (maxCnt - minCnt);
                
                if (type === "keyword") {{
                    const minWidth = 100;
                    const maxWidth = 160;
                    const minHeight = 50;
                    const maxHeight = 80;
                    const width = minWidth + (maxWidth - minWidth) * normalizedCnt;
                    const height = minHeight + (maxHeight - minHeight) * normalizedCnt;
                    return {{ width: width, height: height }};
                }} else if (type === "summary") {{
                    const minWidth = 80;
                    const maxWidth = 140;
                    const minHeight = 35;
                    const maxHeight = 60;
                    const width = minWidth + (maxWidth - minWidth) * normalizedCnt;
                    const height = minHeight + (maxHeight - minHeight) * normalizedCnt;
                    return {{ width: width, height: height }};
                }}
                return {{ width: 100, height: 50 }};
            }}

            function toggleRoot() {{
                treeData.expanded = !treeData.expanded;
                renderMindMap();
            }}

            function toggleKeyword(keywordId) {{
                function findAndToggle(node) {{
                    if (node.id === keywordId) {{
                        node.expanded = !node.expanded;
                        return true;
                    }}
                    if (node.children) {{
                        return node.children.some(findAndToggle);
                    }}
                    return false;
                }}
                findAndToggle(treeData);
                renderMindMap();
            }}

            function showTooltip(event, node) {{
                const tooltip = document.getElementById("tooltip");
                tooltip.style.display = "block";
                tooltip.style.left = event.pageX + 15 + "px";
                tooltip.style.top = event.pageY + 10 + "px";
                
                let content = `<strong>${{node.name}}</strong><br>`;
                content += `응답자 수: ${{node.cnt}}명`;
                
                if (node.type === 'keyword') {{
                    content += `<br>하위 요약: ${{node.children ? node.children.length : 0}}개`;
                }} else if (node.type === 'summary') {{
                    content += `<br>유형: 요약 내용`;
                }}
                
                tooltip.innerHTML = content;
            }}

            function hideTooltip() {{
                document.getElementById("tooltip").style.display = "none";
            }}

            function createCurvedPath(startX, startY, endX, endY) {{
                const midX = startX + (endX - startX) * 0.6;
                return `M ${{startX}} ${{startY}} Q ${{midX}} ${{startY}} ${{endX}} ${{endY}}`;
            }}

            function renderMindMap() {{
                const container = document.getElementById("mindmapContainer");
                const svg = document.getElementById("connectionSvg");
                
                // 기존 노드들과 연결선 제거
                container.querySelectorAll(".keyword-node, .summary-node").forEach(el => el.remove());
                svg.innerHTML = '';
                
                if (!treeData.expanded) return;

                const rootX = 50 + 140; // 루트 노드 오른쪽 끝
                const rootY = container.offsetHeight / 2;
                const keywordStartX = rootX + 60;
                const verticalSpacing = Math.max(80, container.offsetHeight / (treeData.children.length + 1));

                treeData.children.forEach((keyword, index) => {{
                    const keywordY = (index + 1) * verticalSpacing;
                    const keywordSize = getNodeSize(keyword.cnt, 'keyword');
                    
                    // 키워드 노드 위치
                    const keywordX = keywordStartX;

                    // 메인 연결선 그리기
                    const mainPath = createCurvedPath(rootX, rootY, keywordX, keywordY);
                    const mainLine = document.createElementNS("http://www.w3.org/2000/svg", "path");
                    mainLine.setAttribute("d", mainPath);
                    mainLine.setAttribute("class", "main-branch");
                    svg.appendChild(mainLine);

                    // 키워드 노드 생성
                    const keywordNode = document.createElement('div');
                    keywordNode.className = "keyword-node";
                    keywordNode.style.backgroundColor = keyword.color;
                    keywordNode.style.left = keywordX + "px";
                    keywordNode.style.top = (keywordY - keywordSize.height/2) + "px";
                    keywordNode.style.width = keywordSize.width + "px";
                    keywordNode.onclick = () => toggleKeyword(keyword.id);
                    
                    keywordNode.onmouseover = (e) => showTooltip(e, keyword);
                    keywordNode.onmouseout = hideTooltip;
                    
                    keywordNode.innerHTML = `
                        <div style="font-size: ${{Math.min(18, keywordSize.width / keyword.name.length * 1.2)}}px;">
                            ${{keyword.name.length > 15 ? keyword.name.substring(0, 12) + '...' : keyword.name}}
                        </div>
                        <div class="cnt-indicator">${{keyword.cnt}}명</div>
                        <span class="expand-indicator">${{keyword.expanded ? '−' : '+'}}</span>
                    `;
                    
                    container.appendChild(keywordNode);

                    // Summary 노드들 렌더링
                    if (keyword.expanded && keyword.children && keyword.children.length > 0) {{
                        const summaryStartX = keywordX + keywordSize.width + 50;
                        const summarySpacing = Math.max(45, (container.offsetHeight * 0.6) / keyword.children.length);
                        const summaryStartY = keywordY - (keyword.children.length - 1) * summarySpacing / 2;

                        keyword.children.forEach((summary, summaryIndex) => {{
                            const summaryY = summaryStartY + summaryIndex * summarySpacing;
                            const summarySize = getNodeSize(summary.cnt, 'summary');
                            const summaryX = summaryStartX;

                            // Summary 연결선 그리기
                            const subPath = createCurvedPath(
                                keywordX + keywordSize.width, 
                                keywordY, 
                                summaryX, 
                                summaryY
                            );
                            const subLine = document.createElementNS("http://www.w3.org/2000/svg", "path");
                            subLine.setAttribute("d", subPath);
                            subLine.setAttribute("class", "sub-branch");
                            subLine.setAttribute("stroke", keyword.color);
                            svg.appendChild(subLine);

                            // Summary 노드 생성
                            const summaryNode = document.createElement('div');
                            summaryNode.className = "summary-node";
                            summaryNode.style.backgroundColor = summary.color;
                            summaryNode.style.left = summaryX + "px";
                            summaryNode.style.top = (summaryY - summarySize.height/2) + "px";
                            summaryNode.style.opacity = "0.9";
                            
                            // 텍스트 길이에 따라 폰트 크기 조정
                            const fontSize = Math.min(20, Math.max(15, summarySize.width / summary.name.length * 1.5));
                            summaryNode.style.fontSize = fontSize + "px";
                            
                            summaryNode.onmouseover = (e) => showTooltip(e, summary);
                            summaryNode.onmouseout = hideTooltip;
                            
                            summaryNode.innerHTML = `
                                <div style="padding: 4px;">
                                    ${{summary.name.length > 150 ? summary.name.substring(0, 47) + '...' : summary.name}}
                                </div>
                                <div style="font-size: 10px; background: rgba(255,255,255,0.2); padding: 1px 6px; border-radius: 8px; margin-top: 2px;">
                                    ${{summary.cnt}}명
                                </div>
                            `;
                                
                            container.appendChild(summaryNode);
                        }});
                    }}
                }});
                
                // SVG 크기 동적 조정
                svg.setAttribute("width", container.offsetWidth);
                svg.setAttribute("height", container.offsetHeight);
                
                // 컨테이너 크기가 변경되었을 때 스크롤 위치 조정
                if (container.scrollWidth > container.clientWidth) {{
                    container.style.overflowX = "auto";
                }}
            }}

            // 초기 렌더링
            window.onload = function() {{
                renderMindMap();
            }};
            
            // 윈도우 리사이즈 시 다시 렌더링
            window.onresize = function() {{
                setTimeout(renderMindMap, 100);
            }};
        </script>
    </body>
    </html>
    """
    
    return html_code, dynamic_height




def main():
    st.set_page_config(
        page_title="MindMap & RAG Chatbot",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 계층형 마인드맵 + RAG 챗봇 시각화")
    st.markdown("---")
    
    # 파일 업로드
    uploaded_file = st.file_uploader(
        "CSV 파일을 업로드하세요", 
        type=['csv'],
        help="user_id, total_cl, summary, keywords, SPLITTED 컬럼 필요"
    )
    
    if uploaded_file is not None:
        try:
            # CSV 파일 읽기
            df = pd.read_csv(uploaded_file)
            
            # 컬럼 확인
            mindmap_columns = ['user_id', 'total_cl', 'summary', 'keywords', 'SPLITTED']
            has_mindmap_columns = all(col in df.columns for col in mindmap_columns)
            
            
            if not has_mindmap_columns:
                st.error("마인드맵 또는 RAG 기능을 위한 필수 컬럼이 없습니다.")
                st.info("user_id, total_cl, summary, keywords, SPLITTED")
                st.stop()
            
            # 왼쪽/오른쪽 분할 레이아웃
            left_col, right_col = st.columns([1, 1])
            
            # 마인드맵 생성
            if has_mindmap_columns:
                tree_data = create_tree_data_from_csv(df)
                
                with left_col:
                    st.subheader("🗺️ 인터랙티브 마인드맵")
                    st.markdown("*노드를 클릭하여 펼치기/접기*")
                    
                    # 계층형 마인드맵 시각화 - 동적 높이 적용
                    html_code, dynamic_height = create_hierarchical_mindmap_from_data(tree_data)
                    components.html(html_code, height=dynamic_height, scrolling=False)
                    
                    st.caption(f"📏 트리 크기에 따른 동적 높이: {dynamic_height}px")
                    
                    with st.expander("💡 사용법"):
                        st.markdown("""
                        1. **메인 주제 클릭** → 모든 키워드 표시
                        2. **키워드 클릭** → 해당 요약들 표시  
                        3. **마우스 호버** → 상세 정보 표시
                        4. **노드 크기** = 응답자 수 반영
                        5. **높이 자동 조정** = 데이터 크기에 맞춰 최적화
                        """)
                        
            else:
                with left_col:
                    st.info("마인드맵 생성을 위해서는 user_id, total_cl, summary, keywords 컬럼이 필요합니다.")
            
            with right_col:
                st.subheader("📊 데이터 분석")
                
                if has_mindmap_columns:
                    # 기본 정보 메트릭
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("전체 행수", len(df))
                    with col2:
                        filtered_df = df[df.total_cl != 99]
                        st.metric("유효 응답", len(filtered_df))
                    with col3:
                        st.metric("총 응답자", df.user_id.nunique())
                    
                    # Summary Table
                    st.subheader("📋 Summary Table")
                    summary_table = filtered_df.groupby(['keywords','summary'], as_index=False, dropna=False).agg({'user_id': 'nunique'}).rename(columns={'user_id': 'cnt'})
                    st.dataframe(
                        summary_table.sort_values('cnt', ascending=False), 
                        use_container_width=True,
                        height=200
                    )

            file_hash = get_file_hash(uploaded_file)
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"{file_hash}.csv")

            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        
            rag_chain = initialize_components(temp_path, "gpt-4o-mini")
            chat_history = StreamlitChatMessageHistory(key="chat_messages_user")
        
            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain,
                lambda session_id: chat_history,
                input_messages_key="input",
                history_messages_key="history",
                output_messages_key="answer",
            )
        
            if len(chat_history.messages) == 0:
                chat_history.add_ai_message("업로드된 유저 응답 기반으로 무엇이든 물어보세요! 🤗")
        
            for msg in chat_history.messages:
                st.chat_message(msg.type).write(msg.content)
        
            if prompt_message := st.chat_input("질문을 입력하세요"):
                st.chat_message("human").write(prompt_message)
                with st.chat_message("ai"):
                    with st.spinner("생각 중입니다..."):
                        config = {"configurable": {"session_id": "user_session"}}
                        response = conversational_rag_chain.invoke(
                            {"input": prompt_message},
                            config,
                        )
                        answer = response['answer']
                        st.write(answer)
        
                        if "관련된 내용이 없습니다" not in answer and response.get("context"):
                            with st.expander("참고 문서 확인"):
                                for doc in response['context']:
                                    source = doc.metadata.get('source', '알 수 없음')
                                    source_filename = os.path.basename(source)
                                    st.markdown(f"👤 {source_filename}")
                                    st.markdown(doc.page_content)
        
                    
        except Exception as e:
            st.error(f"파일 처리 중 오류가 발생했습니다: {str(e)}")
            st.exception(e)
    
    else:
        # 샘플 정보 표시
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.info("💡 CSV 파일을 업로드하면 데이터 기반 마인드맵과 RAG 챗봇을 사용할 수 있습니다.")
            
            with st.expander("🎨 계층형 마인드맵의 특징"):
                st.markdown("""
                **🏗️ 구조**
                - 메인 주제가 왼쪽에 위치
                - 키워드들이 오른쪽으로 확장 (세로 배열)
                - 요약들이 각 키워드에서 더 확장
                - 곡선 연결선으로 자연스러운 연결
                
                **🎯 인터랙션**  
                - 메인 주제 클릭 → 모든 키워드 표시
                - 키워드 클릭 → 해당 요약들 표시
                - 노드 크기 = 응답자 수 반영
                - 마우스 호버 → 상세 정보 표시
                """)
        
        with col2:
            with st.expander("📋 CSV 파일 형식 요구사항"):
                st.markdown("""
                **마인드맵용 (필수):**
                ```
                user_id, total_cl, summary, keywords
                user001, 1, "제품이 만족스럽다", "제품 만족도"
                user002, 2, "가격이 합리적이다", "가격"
                user003, 99, "무효 응답", ""
                ```
                
                **RAG 챗봇용 (선택):**
                ```
                user_id, answer
                user001, "제품에 대한 상세한 의견..."
                user002, "서비스 경험에 대한 설명..."
                ```
                
                * total_cl != 99 인 데이터만 마인드맵에 사용됩니다
                * 두 기능을 모두 사용하려면 모든 컬럼이 필요합니다
                """)

if __name__ == "__main__":
    main()
