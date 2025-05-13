import streamlit as st
import hmac
st.set_page_config(
    page_title="AI-Консультант",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded"
)
from streamlit_extras.badges import badge
from streamlit_extras.metric_cards import style_metric_cards
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
import os
import tempfile
import gdown
import re
import zipfile
import shutil
import logging
import time
import datetime
import hashlib
from functools import lru_cache
import matplotlib.pyplot as plt
import numpy as np
from numpy import dot
from numpy.linalg import norm

# --- Аутентификация по паролю через Streamlit secrets ---
def check_password():
    """Returns True if the user entered the correct password."""
    if "password_correct" in st.session_state:
        return st.session_state.password_correct
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state.password, st.secrets.get("PASSWORD", "default_password")):
            st.session_state.password_correct = True
            del st.session_state.password  # Не храним пароль в сессии
        else:
            st.session_state.password_correct = False
    
    st.text_input(
        "Введите пароль для доступа к приложению", 
        type="password",
        key="password",
        on_change=password_entered
    )
    
    if "password_correct" in st.session_state:
        if not st.session_state.password_correct:
            st.error("😕 Неверный пароль")
            return False
    
    return False

# Если пароль неверный, остановить выполнение приложения
if not check_password():
    st.stop()

# Оформление интерфейса Streamlit
st.markdown("""
<style>
    .relevance-high {
        color: green;
        font-size: 0.8rem;
        text-align: right;
    }
    .relevance-medium {
        color: orange;
        font-size: 0.8rem;
        text-align: right;
    }
    .relevance-low {
        color: red;
        font-size: 0.8rem;
        text-align: right;
    }
    .source-info {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 10px;
        margin-top: 10px;
        border-left: 3px solid #4CAF50;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.2rem;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 0.8rem;
    }
    .metadata {
        font-size: 0.8rem;
        color: #666;
        text-align: right;
    }
    .base-knowledge {
        border-left: 3px solid #4CAF50;
        padding-left: 10px;
        background-color: #f0f8f0;
    }
    .additional-knowledge {
        border-left: 3px solid #FFA500;
        padding-left: 10px;
        background-color: #fff8f0;
    }
    .no-knowledge {
        border-left: 3px solid #FF0000;
        padding-left: 10px;
        background-color: #fff0f0;
    }
    .source-header {
        font-weight: bold;
        margin-top: 10px;
    }
    .relevance-bar {
        height: 8px;
        border-radius: 4px;
        margin-bottom: 5px;
    }
    .mode-badge {
        padding: 5px 10px;
        border-radius: 10px;
        font-weight: bold;
        font-size: 0.8rem;
        display: inline-block;
    }
    .mode-strict {
        background-color: #ffcccc;
        color: #990000;
    }
    .mode-balanced {
        background-color: #fff2cc;
        color: #806600;
    }
    .mode-flexible {
        background-color: #d9ead3;
        color: #274e13;
    }
</style>
""", unsafe_allow_html=True)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger('ai-consultant')

# Инициализация session_state для хранения данных
if "faiss_db" not in st.session_state:
    st.session_state.faiss_db = None
if "faiss_path" not in st.session_state:
    st.session_state.faiss_path = None
if "doc_count" not in st.session_state:
    st.session_state.doc_count = 0
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_context" not in st.session_state:
    st.session_state.conversation_context = ""
if "model" not in st.session_state:
    st.session_state.model = "gpt-4o-mini"
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.4
if "max_token" not in st.session_state:
    st.session_state.max_token = 2000
if "use_chunk" not in st.session_state:
    st.session_state.use_chunk = 4
if "temp_dir" not in st.session_state:
    st.session_state.temp_dir = tempfile.mkdtemp()
if "knowledge_mode" not in st.session_state:
    st.session_state.knowledge_mode = "Сбалансированный"
if "show_sources" not in st.session_state:
    st.session_state.show_sources = False
if "search_analytics" not in st.session_state:
    st.session_state.search_analytics = {
        "queries": [],
        "no_results_queries": [],
        "avg_docs_found": 0,
        "total_docs_found": 0,
        "query_count": 0
    }
if "max_context_turns" not in st.session_state:
    st.session_state.max_context_turns = 10

# Загрузка API-ключа из секретов Streamlit
api_key = st.secrets.get("OPENAI_API_KEY")

# Проверка наличия API-ключа
if not api_key:
    st.error("❌ Ошибка: API-ключ OpenAI не найден!")
    st.stop()

# Инициализация клиента OpenAI
client = OpenAI(api_key=api_key)

def validate_google_drive_url(url):
    """Функция для валидации ссылки Google Drive."""
    if 'file' in url:
        patterns = [
            r'drive\.google\.com/file/d/([^/]+)',
            r'drive\.google\.com/open\?id=([^&]+)'
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return True, match.group(1), 'file'
    if 'folders' in url:
        pattern = r'drive\.google\.com/drive/folders/([^/?&#]+)'
        match = re.search(pattern, url)
        if match:
            return True, match.group(1), 'folder'
    return False, None, None

def download_file_from_drive(file_id, output_path):
    """Загрузка файла из Google Drive."""
    try:
        with st.spinner("Загрузка файла из Google Drive..."):
            direct_url = f"https://drive.google.com/uc?id={file_id}&export=download"
            gdown.download(direct_url, output_path, quiet=True)
            return output_path
    except Exception as e:
        st.error(f"Ошибка при загрузке файла из Google Drive: {str(e)}")
        logger.error(f"Ошибка при загрузке файла из Google Drive: {str(e)}")
        return None

def download_from_drive_folder(folder_id, output_dir):
    """Загрузка файлов из папки Google Drive."""
    try:
        with st.spinner("Загрузка файлов из папки Google Drive..."):
            folder_url = f"https://drive.google.com/drive/folders/{folder_id}"
            files = gdown.download_folder(
                folder_url,
                output=output_dir,
                quiet=True,
                use_cookies=False
            )
            return files
    except Exception as e:
        st.error(f"Ошибка при загрузке файлов из папки Google Drive: {str(e)}")
        logger.error(f"Ошибка при загрузке файлов из папки Google Drive: {str(e)}")
        return []

def load_faiss_db(faiss_path, embeddings=None):
    """Загрузка FAISS базы данных."""
    try:
        if embeddings is None:
            embeddings = OpenAIEmbeddings()
        db = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
        doc_count = len(db.index_to_docstore_id) if hasattr(db, 'index_to_docstore_id') else "неизвестно"
        st.session_state.faiss_db = db
        st.session_state.faiss_path = faiss_path
        st.session_state.doc_count = doc_count
        logger.info(f"База FAISS успешно загружена. Документов: {doc_count}")
        return db, doc_count
    except Exception as e:
        st.error(f"❌ Ошибка загрузки FAISS: {e}")
        logger.error(f"Ошибка загрузки FAISS: {e}")
        return None, 0

def get_system_message(knowledge_mode, has_relevant_docs, conversation_context_exists):
    """Получение системного сообщения в зависимости от режима работы."""
    base_message = "Ты — AI-ассистент, помогающий пользователям с вопросами на основе базы знаний. "
    
    mode_instructions = {
        "Строгий": """
            Ты работаешь в СТРОГОМ режиме. Это означает:
            1. Отвечай ИСКЛЮЧИТЕЛЬНО на основе информации из предоставленного контекста.
            2. Если в контексте нет информации — четко скажи: 'В базе знаний нет информации по этому вопросу'.
            3. НЕ ИСПОЛЬЗУЙ никакие внешние знания, даже если они кажутся очевидными.
            4. НИКОГДА не пытайся угадать или предположить ответ при отсутствии данных.
            5. Точность важнее полноты — лучше дать неполный, но точный ответ.
        """,
        "Сбалансированный": """
            Ты работаешь в СБАЛАНСИРОВАННОМ режиме. Это означает:
            1. Приоритет информации из контекста базы знаний.
            2. Если информация в контексте неполная — можешь дополнить её, но чётко отметь.
            3. Если в контексте нет информации — скажи об этом, но можешь предложить общий ответ.
            4. Старайся найти баланс между точностью (из базы) и полнотой (общие знания).
        """,
        "Гибкий": """
            Ты работаешь в ГИБКОМ режиме. Это означает:
            1. Используй информацию из контекста как основу, но смело дополняй её.
            2. Если в контексте есть частичная информация — расширь её своими знаниями.
            3. Даже при отсутствии информации в контексте старайся дать полезный ответ.
            4. Приоритет — максимальная полезность ответа для пользователя.
            5. Отмечай, где заканчивается информация из базы знаний.
        """
    }
    
    search_status = (
        "ВАЖНО: В базе знаний НЕ НАЙДЕНО прямой информации по запросу пользователя. "
        if not has_relevant_docs else
        "База знаний содержит информацию, относящуюся к запросу пользователя."
    )
    
    dialog_status = (
        "Текущий вопрос может быть связан с предыдущим контекстом разговора."
        if conversation_context_exists else
        "Это начало нового разговора или новая тема."
    )
    
    final_instructions = """
        При ответе на вопрос:
        1. Анализируй контекст полностью перед ответом.
        2. Структурируй информацию логически.
        3. Используй списки и выделение для лучшей читаемости.
        4. Если вопрос многосоставный, ответь на все его части.
        5. Твоя цель — максимально полезный и точный ответ в рамках выбранного режима работы.
    """
    
    return f"{base_message}\n\n{mode_instructions[knowledge_mode]}\n\n{search_status}\n\n{dialog_status}\n\n{final_instructions}"

@lru_cache(maxsize=50)
def cached_search(query_hash, k, knowledge_mode):
    """Кэшированный поиск в базе знаний."""
    try:
        results = st.session_state.faiss_db.similarity_search_with_score(query_hash, k=k)
        
        thresholds = {
            "Строгий": 0.8,
            "Сбалансированный": 0.6,
            "Гибкий": 0.4
        }
        threshold = thresholds.get(knowledge_mode, 0.5)
        
        filtered_results = [(doc, score) for doc, score in results if score < threshold]
        return filtered_results, len(filtered_results) > 0
    except Exception as e:
        logger.error(f"Ошибка при выполнении кэшированного поиска: {e}")
        return [], False

def estimate_tokens(text):
    """Оценка количества токенов в тексте."""
    return len(text.split()) * 1.5

def update_conversation_context():
    """Обновление контекста разговора."""
    relevant_messages = st.session_state.messages[-st.session_state.max_context_turns*2:] if len(st.session_state.messages) > st.session_state.max_context_turns*2 else st.session_state.messages
    
    context_parts = []
    for i in range(0, len(relevant_messages), 2):
        if i+1 < len(relevant_messages):
            user_msg = relevant_messages[i]["content"]
            assistant_msg = relevant_messages[i+1]["content"]
            context_parts.append(f"Пользователь: {user_msg}\nАссистент: {assistant_msg}")
    
    st.session_state.conversation_context = "\n\n".join(context_parts)
    logger.info(f"Обновлен контекст разговора. Размер: {len(st.session_state.conversation_context)} символов")

def answer_query(query):
    """Обработка запроса пользователя."""
    start_time = time.time()
    
    try:
        # Проверка на запросы о возможностях
        general_questions = ["что ты умеешь", "помощь", "справка", "возможности", "команды"]
        if any(q in query.lower() for q in general_questions):
            capabilities = f"""
            # Возможности AI-консультанта

            Я AI-консультант с функцией анализа базы знаний. Я могу:

            1. Искать информацию в заданной базе знаний по вашим вопросам
            2. Отвечать с учетом контекста предыдущих ответов для уточняющих вопросов
            3. Работать в трех режимах:
               - **Строгий режим**: отвечаю ТОЛЬКО на основе информации из базы знаний
               - **Сбалансированный режим**: приоритет информации из базы знаний с дополнениями
               - **Гибкий режим**: использую базу знаний с расширенными дополнениями
            
            Текущий режим: **{st.session_state.knowledge_mode}**
            
            Задайте мне вопрос, и я постараюсь на него ответить!
            """
            logger.info(f"Запрос о возможностях. Время: {time.time() - start_time:.2f}с")
            return capabilities, {"relevance": 1.0, "docs_count": 0, "query_time": time.time() - start_time}
        
        # Проверка наличия базы данных
        if not st.session_state.faiss_db:
            logger.warning("Попытка запроса без загруженной базы данных")
            return "❌ База данных не загружена. Пожалуйста, загрузите FAISS базу в боковом меню.", {"relevance": 0, "docs_count": 0, "query_time": time.time() - start_time}
        
        # Поиск в базе знаний
        logger.info(f"Поиск для запроса: '{query}'")
        with st.spinner("🔍 Поиск релевантной информации..."):
            docs_with_scores, has_relevant_docs = cached_search(query, st.session_state.use_chunk, st.session_state.knowledge_mode)
        
        # Обработка результатов поиска
        docs = [doc for doc, _ in docs_with_scores]
        scores = [score for _, score in docs_with_scores]
        avg_relevance = sum(scores) / len(scores) if scores else 1.0
        rel_score = 1 / (1 + avg_relevance)  # 0..1, где 1 — идеально
        
        # Обновление метрик
        st.session_state.search_analytics["queries"].append(query)
        st.session_state.search_analytics["query_count"] += 1
        st.session_state.search_analytics["total_docs_found"] += len(docs)
        st.session_state.search_analytics["avg_docs_found"] = (
            st.session_state.search_analytics["total_docs_found"] / 
            st.session_state.search_analytics["query_count"]
        )
        
        if not has_relevant_docs:
            st.session_state.search_analytics["no_results_queries"].append(query)
        
        # Формирование контекста
        context = "\n\n".join([doc.page_content for doc in docs])
        max_context_tokens = 4000 if "gpt-4" in st.session_state.model else 3000
        
        if st.session_state.conversation_context:
            enhanced_context = st.session_state.conversation_context + "\n\n" + context
        else:
            enhanced_context = context
        
        # Ограничение размера контекста
        while estimate_tokens(enhanced_context) > max_context_tokens and len(docs) > 1:
            docs = docs[:-1]
            context = "\n\n".join([doc.page_content for doc in docs])
            enhanced_context = st.session_state.conversation_context + "\n\n" + context if st.session_state.conversation_context else context
        
        # Получение системного сообщения
        system_message = get_system_message(
            st.session_state.knowledge_mode,
            has_relevant_docs,
            bool(st.session_state.conversation_context)
        )
        
        # Генерация ответа
        with st.spinner("🧠 Генерация ответа..."):
            messages = [
                {"role": "system", "content": system_message},
                *st.session_state.messages,
                {"role": "user", "content": f"Контекст для ответа:\n\n{enhanced_context}\n\nВопрос пользователя: {query}"}
            ]
            
            response = client.chat.completions.create(
                model=st.session_state.model,
                messages=messages,
                temperature=st.session_state.temperature,
                max_tokens=st.session_state.max_token,
            )
            
            answer = response.choices[0].message.content
        
        # Форматирование ответа
        if st.session_state.knowledge_mode in ["Сбалансированный", "Гибкий"]:
            if "На основе базы знаний" not in answer and "Согласно базе" not in answer and has_relevant_docs:
                base_pattern = r'(.*?)(?:Дополнительно|Однако|Впрочем| могу добавить| стоит отметить)'
                match = re.search(base_pattern, answer, re.DOTALL)
                
                if match:
                    base_part = match.group(1).strip()
                    additional_part = answer[len(base_part):].strip()
                    
                    formatted_answer = f"<div class='base-knowledge'>{base_part}</div>\n\n"
                    if additional_part:
                        formatted_answer += f"<div class='additional-knowledge'>{additional_part}</div>"
                    
                    answer = formatted_answer
        
        if not has_relevant_docs and st.session_state.knowledge_mode == "Строгий":
            answer = f"<div class='no-knowledge'>{answer}</div>"
        
        # Обновление контекста и метрик
        if not st.session_state.conversation_context:
            st.session_state.conversation_context = answer
        else:
            st.session_state.conversation_context += f"\n\nПользователь спросил: {query}\n\nОтвет: {answer}"
        
        query_time = time.time() - start_time
        
        logger.info(f"Запрос обработан за {query_time:.2f}с, Релевантность: {rel_score:.4f}")
        
        # Сериализуемые данные для источников
        docs_data = [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata if hasattr(doc, 'metadata') else {}
            }
            for doc in docs
        ]
        
        return answer, {
            "relevance": rel_score,
            "docs_count": len(docs),
            "docs_data": docs_data,  # Используем сериализуемую версию
            "query_time": query_time,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode": st.session_state.knowledge_mode
        }
        
    except Exception as e:
        error_message = f"❌ Ошибка при обработке запроса: {str(e)}"
        logger.error(error_message)
        return error_message, {"relevance": 0, "docs_count": 0, "query_time": time.time() - start_time}

# --- SIDEBAR: Загрузка базы и настройки ---
with st.sidebar:
    st.header("💾 Загрузка FAISS базы данных")
    upload_method = st.radio("Выберите источник FAISS базы:", ["Локальный компьютер", "Google Drive"], horizontal=True)
    
    if upload_method == "Локальный компьютер":
        col1, col2 = st.columns(2)
        with col1:
            faiss_file = st.file_uploader("Загрузите index.faiss файл", type=["faiss"])
        with col2:
            pkl_file = st.file_uploader("Загрузите index.pkl файл", type=["pkl"])
        if faiss_file and pkl_file:
            faiss_dir = os.path.join(st.session_state.temp_dir, "faiss_index")
            os.makedirs(faiss_dir, exist_ok=True)
            with open(os.path.join(faiss_dir, "index.faiss"), "wb") as f:
                f.write(faiss_file.getbuffer())
            with open(os.path.join(faiss_dir, "index.pkl"), "wb") as f:
                f.write(pkl_file.getbuffer())
            if st.button("Загрузить FAISS базу"):
                with st.spinner("Загружаем базу..."):
                    embeddings = OpenAIEmbeddings()
                    db, doc_count = load_faiss_db(faiss_dir, embeddings)
                    if db:
                        st.success(f"✅ База данных успешно загружена! Документов: {doc_count}")
    else:  # Google Drive
        st.subheader("Загрузка с Google Drive по прямой ссылке")
        drive_url = st.text_input("Ссылка на файл или папку в Google Drive")
        if drive_url:
            if st.button("Загрузить с Google Drive"):
                is_valid, id_value, url_type = validate_google_drive_url(drive_url)
                if is_valid:
                    if url_type == 'file':
                        if '.zip' in drive_url.lower():
                            with st.spinner("Загружаем и распаковываем архив..."):
                                zip_path = os.path.join(st.session_state.temp_dir, "faiss_base.zip")
                                zip_success = download_file_from_drive(id_value, zip_path)
                                if zip_success:
                                    extract_dir = os.path.join(st.session_state.temp_dir, "extracted_faiss")
                                    os.makedirs(extract_dir, exist_ok=True)
                                    try:
                                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                                            zip_ref.extractall(extract_dir)
                                        faiss_file = os.path.join(extract_dir, "index.faiss")
                                        pkl_file = os.path.join(extract_dir, "index.pkl")
                                        if os.path.exists(faiss_file) and os.path.exists(pkl_file):
                                            faiss_dir = os.path.join(st.session_state.temp_dir, "faiss_index")
                                            os.makedirs(faiss_dir, exist_ok=True)
                                            shutil.copy2(faiss_file, os.path.join(faiss_dir, "index.faiss"))
                                            shutil.copy2(pkl_file, os.path.join(faiss_dir, "index.pkl"))
                                            embeddings = OpenAIEmbeddings()
                                            db, doc_count = load_faiss_db(faiss_dir, embeddings)
                                            if db:
                                                st.success(f"✅ База данных успешно загружена из архива! Документов: {doc_count}")
                                        else:
                                            st.error("В архиве не найдены необходимые файлы index.faiss и index.pkl")
                                    except Exception as e:
                                        st.error(f"Ошибка при распаковке архива: {e}")
                                else:
                                    st.error("Ошибка при загрузке архива с Google Drive")
                        else:
                            st.warning("Ссылка должна указывать на ZIP-архив с базой или на папку, содержащую файлы базы")
                    elif url_type == 'folder':
                        with st.spinner("Загружаем файлы из папки..."):
                            output_dir = os.path.join(st.session_state.temp_dir, "drive_folder")
                            os.makedirs(output_dir, exist_ok=True)
                            files = download_from_drive_folder(id_value, output_dir)
                            if files:
                                faiss_file = None
                                pkl_file = None
                                for file in files:
                                    if file.endswith("index.faiss"):
                                        faiss_file = file
                                    elif file.endswith("index.pkl"):
                                        pkl_file = file
                                if faiss_file and pkl_file:
                                    faiss_dir = os.path.join(st.session_state.temp_dir, "faiss_index")
                                    os.makedirs(faiss_dir, exist_ok=True)
                                    shutil.copy2(faiss_file, os.path.join(faiss_dir, "index.faiss"))
                                    shutil.copy2(pkl_file, os.path.join(faiss_dir, "index.pkl"))
                                    embeddings = OpenAIEmbeddings()
                                    db, doc_count = load_faiss_db(faiss_dir, embeddings)
                                    if db:
                                        st.success(f"✅ База данных успешно загружена из папки! Документов: {doc_count}")
                                else:
                                    st.error("В папке не найдены необходимые файлы index.faiss и index.pkl")
                            else:
                                st.error("Ошибка при загрузке файлов из папки Google Drive")
                else:
                    st.error("Неверный формат ссылки Google Drive")
    
    # Режим работы
    st.header("Режим работы")
    st.session_state.knowledge_mode = st.select_slider(
        "Режим использования базы знаний:",
        options=["Гибкий", "Сбалансированный", "Строгий"],
        value=st.session_state.knowledge_mode,
        help="Гибкий - использует базу как дополнение, Сбалансированный - приоритет базе знаний, Строгий - только информация из базы"
    )

    # Экспорт диалога
    if st.button("🧹 Очистить историю чата"):
        st.session_state.messages = []
        st.session_state.conversation_context = ""
        st.rerun()

    # Кнопка экспорта диалога теперь после очистки чата
    if st.button("💾 Экспорт диалога"):
        chat_export = "\n\n".join([
            f"{'Пользователь' if msg['role'] == 'user' else 'Ассистент'}: {msg['content']}"
            for msg in st.session_state.messages
        ])
        st.download_button(
            label="Скачать диалог",
            data=chat_export,
            file_name=f"chat_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain"
        )

    # --- Новые настройки модели и генерации ---
    st.header("⚙️ Настройки генерации")
    st.session_state.model = st.selectbox("Выберите модель:", ["gpt-4o-mini", "gpt-4o", "gpt-4.1"], index=0)
    st.session_state.temperature = st.slider("Температура генерации:", 0.0, 2.0, 0.4, 0.1)
    st.session_state.max_token = st.slider("Максимальное число токенов:", 1500, 4000, 2000, 100)
    st.session_state.use_chunk = st.slider("Число блоков контекста:", 4, 10, 4, 1)

# --- MAIN: Чат ---
st.title("AI-Консультант")

# Статус сессии
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"**Режим:** {st.session_state.knowledge_mode}")
with col2:
    st.markdown(f"**Документов в базе:** {st.session_state.doc_count}")
with col3:
    if st.session_state.faiss_db:
        st.markdown("**Статус:** ✅ Активен")
    else:
        st.markdown("**Статус:** ⏳ Ожидание базы")

# Чат
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# Обработка нового запроса
if prompt := st.chat_input("Введите ваш запрос..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Отображаем статус в отдельном блоке (не внутри сообщения)
    status_container = st.empty()
    with status_container.status("🔍 Поиск информации...", expanded=True) as status:
        status.update(label="Поиск в базе знаний...", state="running")
        answer, metadata = answer_query(prompt)
        status.update(label="Готово!", state="complete")
    # Скрываем статус после завершения
    status_container.empty()
    
    # Отображаем ответ в чате
    with st.chat_message("assistant"):
        st.markdown(answer, unsafe_allow_html=True)
        relevance_color = "green" if metadata["relevance"] > 0.8 else "orange" if metadata["relevance"] > 0.5 else "red"
        st.markdown(
            f"<div style='text-align: right'><small style='color: {relevance_color}'>"
            f"Релевантность: {metadata['relevance']:.0%}</small></div>",
            unsafe_allow_html=True
        )
    # Добавляем ответ в историю сообщений
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": metadata.get("docs_data", [])  # Теперь это JSON-сериализуемые словари
    })
