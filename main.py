import os
import tempfile
import shutil
from pathlib import Path
from typing import Optional
import yt_dlp
import whisper
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import aiofiles
import uuid
import re
import json
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel

# 載入環境變數
load_dotenv()

app = FastAPI(title="YouTube 文字轉錄器")

# API 設定相關的資料模型
class ApiKeyRequest(BaseModel):
    api_key: str

# API 金鑰儲存檔案路徑
API_KEY_FILE = "api_key.json"

# 建立必要的目錄
os.makedirs("temp", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# 設定模型路徑到 D 槽
os.environ["XDG_CACHE_HOME"] = "D:\\whisper_models"

# 全域模型變數，動態載入
current_model = None
current_model_name = None

# 進度追蹤
progress_data = {
    "transcription": {"progress": 0, "status": "", "message": ""},
    "translation": {"progress": 0, "status": "", "message": ""}
}

def update_progress(module: str, progress: int, status: str = "", message: str = ""):
    """更新進度"""
    global progress_data
    progress_data[module] = {
        "progress": progress,
        "status": status,
        "message": message
    }
    print(f"📊 [{module.upper()}] {progress}% - {status}: {message}")

def load_whisper_model(model_name: str):
    """動態載入 Whisper 模型"""
    global current_model, current_model_name
    
    if current_model_name == model_name:
        print(f"✅ 模型 {model_name} 已載入，無需重新載入")
        return current_model
    
    print(f"🔄 正在載入 Whisper 模型: {model_name}...")
    current_model = whisper.load_model(model_name)
    current_model_name = model_name
    print(f"✅ Whisper 模型 {model_name} 載入完成！")
    return current_model

def save_api_key(api_key: str):
    """儲存 API 金鑰到檔案"""
    try:
        with open(API_KEY_FILE, 'w', encoding='utf-8') as f:
            json.dump({"api_key": api_key}, f)
        print("✅ API 金鑰已儲存")
        return True
    except Exception as e:
        print(f"❌ 儲存 API 金鑰失敗: {str(e)}")
        return False

def load_api_key():
    """從檔案載入 API 金鑰"""
    try:
        if os.path.exists(API_KEY_FILE):
            with open(API_KEY_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("api_key", "")
        return ""
    except Exception as e:
        print(f"❌ 載入 API 金鑰失敗: {str(e)}")
        return ""

def test_openai_api(api_key: str):
    """測試 OpenAI API 金鑰是否有效"""
    try:
        test_client = OpenAI(api_key=api_key)
        # 發送一個簡單的測試請求
        response = test_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        return True, "API 金鑰有效"
    except Exception as e:
        error_str = str(e)
        
        # 處理常見的錯誤類型
        if "insufficient_quota" in error_str or "quota" in error_str.lower():
            return False, "API 金鑰有效，但配額不足。請檢查您的 OpenAI 帳戶餘額和計費設定。"
        elif "invalid_api_key" in error_str or "authentication" in error_str.lower():
            return False, "API 金鑰無效，請檢查金鑰是否正確。"
        elif "rate_limit" in error_str.lower():
            return False, "API 金鑰有效，但請求頻率過高，請稍後再試。"
        else:
            return False, f"API 測試失敗: {error_str}"

# 靜態檔案和模板
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# OpenAI 客戶端初始化
openai_client = None

def initialize_openai_client():
    """初始化 OpenAI 客戶端"""
    global openai_client
    try:
        # 優先使用儲存的 API 金鑰
        saved_api_key = load_api_key()
        if saved_api_key:
            openai_client = OpenAI(api_key=saved_api_key)
            print("✅ OpenAI API 客戶端初始化成功 (使用儲存的 API 金鑰)")
            return True
        
        # 其次使用環境變數
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            openai_client = OpenAI(api_key=openai_api_key)
            print("✅ OpenAI API 客戶端初始化成功 (使用環境變數)")
            return True
        
        print("⚠️ 未找到 OpenAI API 金鑰，翻譯功能將無法使用")
        return False
    except Exception as e:
        print(f"❌ OpenAI API 客戶端初始化失敗: {str(e)}")
        return False

# 初始化 OpenAI 客戶端
initialize_openai_client()

def validate_youtube_url(url: str) -> bool:
    """驗證 YouTube URL 是否有效"""
    youtube_regex = re.compile(
        r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/'
        r'(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
    )
    return youtube_regex.match(url) is not None

def download_audio(url: str, output_dir: str) -> str:
    """使用 yt-dlp 下載 YouTube 音訊"""
    update_progress("transcription", 10, "開始下載", "正在連接 YouTube...")
    
    ydl_opts = {
        'format': 'bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio/best',
        'outtmpl': os.path.join(output_dir, 'audio.%(ext)s'),
        'noplaylist': True,
        'no_warnings': False,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # 先提取資訊
            update_progress("transcription", 20, "提取資訊", "正在提取影片資訊...")
            info = ydl.extract_info(url, download=False)
            title = info.get('title', 'unknown')
            duration = info.get('duration', 0)
            print(f"📺 影片標題: {title}")
            print(f"⏱️ 影片長度: {duration//60}分{duration%60}秒")
            
            # 下載音訊
            update_progress("transcription", 40, "下載中", "正在下載音訊檔案...")
            ydl.download([url])
            update_progress("transcription", 70, "下載完成", "音訊下載完成！")
            
            # 尋找下載的檔案
            update_progress("transcription", 80, "檢查檔案", "正在檢查下載的檔案...")
            downloaded_files = []
            print(f"📁 檢查目錄: {output_dir}")
            print(f"📂 目錄內容: {os.listdir(output_dir)}")
            
            for file in os.listdir(output_dir):
                if file.endswith(('.wav', '.mp3', '.m4a', '.webm', '.mp4')):
                    file_path = os.path.join(output_dir, file)
                    file_size = os.path.getsize(file_path)
                    downloaded_files.append(file_path)
                    print(f"🎵 找到音訊檔案: {file_path}")
                    print(f"📊 檔案大小: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
                    print(f"✅ 檔案存在檢查: {os.path.exists(file_path)}")
            
            if downloaded_files:
                # 返回第一個找到的檔案
                selected_file = downloaded_files[0]
                print(f"🎯 選擇檔案: {selected_file}")
                update_progress("transcription", 100, "完成", "音訊下載完成！")
                return selected_file
            else:
                update_progress("transcription", 0, "錯誤", "找不到下載的音訊檔案")
                raise Exception("找不到下載的音訊檔案")
            
    except Exception as e:
        update_progress("transcription", 0, "錯誤", f"下載音訊失敗: {str(e)}")
        raise Exception(f"下載音訊失敗: {str(e)}")

def transcribe_audio(audio_path: str, model_name: str = "large-v3") -> dict:
    """使用 Whisper 轉錄音訊並翻譯成英文"""
    update_progress("transcription", 0, "開始轉錄", "正在準備轉錄...")
    
    try:
        # 檢查檔案是否存在
        if not os.path.exists(audio_path):
            update_progress("transcription", 0, "錯誤", f"音訊檔案不存在: {audio_path}")
            raise Exception(f"音訊檔案不存在: {audio_path}")
        
        # 使用絕對路徑
        abs_path = os.path.abspath(audio_path)
        file_size = os.path.getsize(abs_path)
        print(f"📁 轉錄檔案: {abs_path}")
        print(f"📊 檔案大小: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
        
        # 確保 FFmpeg 路徑在環境變數中
        ffmpeg_path = r"C:\Users\NITRO\Downloads\ffmpeg-8.0-essentials_build\ffmpeg-8.0-essentials_build\bin"
        if ffmpeg_path not in os.environ.get("PATH", ""):
            os.environ["PATH"] = ffmpeg_path + ";" + os.environ.get("PATH", "")
            print(f"🔧 已添加 FFmpeg 路徑到環境變數")
        
        # 載入指定的模型
        update_progress("transcription", 10, "載入模型", f"正在載入 {model_name} 模型...")
        model = load_whisper_model(model_name)
        
        # 使用 Whisper 進行轉錄和翻譯，指定使用 FP32
        update_progress("transcription", 20, "開始轉錄", "正在分析音訊內容...")
        print("🤖 開始 Whisper 轉錄處理...")
        print("⏳ 這可能需要幾分鐘時間，請耐心等待...")
        print("📝 正在分析音訊內容...")
        
        result = model.transcribe(
            abs_path, 
            language=None,  # 自動偵測語言
            task="translate",  # 翻譯成英文
            fp16=False,  # 強制使用 FP32，避免 CPU 上的 FP16 問題
            verbose=True  # 顯示詳細進度
        )
        
        update_progress("transcription", 90, "轉錄完成", "Whisper 轉錄處理完成！")
        print("✅ Whisper 轉錄處理完成！")
        print(f"📊 轉錄結果: 共 {len(result['segments'])} 個段落")
        update_progress("transcription", 100, "完成", "音訊轉錄完成！")
        return result
    except Exception as e:
        update_progress("transcription", 0, "錯誤", f"轉錄失敗: {str(e)}")
        print(f"❌ 轉錄錯誤詳情: {str(e)}")
        print(f"📁 檔案路徑: {audio_path}")
        print(f"📁 絕對路徑: {os.path.abspath(audio_path) if os.path.exists(audio_path) else '檔案不存在'}")
        print(f"🔧 當前 PATH: {os.environ.get('PATH', '')[:200]}...")
        raise Exception(f"轉錄失敗: {str(e)}")

def translate_text_with_openai(text: str, target_language: str) -> str:
    """使用 OpenAI API 翻譯文字"""
    if not openai_client:
        raise Exception("OpenAI API 客戶端未初始化，請檢查 API 金鑰")
    
    update_progress("translation", 20, "開始翻譯", f"正在翻譯到 {target_language}...")
    
    # 語言代碼對應
    language_names = {
        'zh-TW': '繁體中文',
        'zh-CN': '簡體中文', 
        'en': '英文',
        'ja': '日文',
        'ko': '韓文',
        'es': '西班牙文',
        'fr': '法文',
        'de': '德文'
    }
    
    target_lang_name = language_names.get(target_language, target_language)
    
    try:
        update_progress("translation", 50, "翻譯中", "正在呼叫 OpenAI API...")
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": f"你是一個專業的翻譯專家。請將以下文字翻譯成{target_lang_name}，保持原文的語氣和風格，確保翻譯準確自然。"
                },
                {
                    "role": "user", 
                    "content": text
                }
            ],
            max_tokens=4000,
            temperature=0.3
        )
        
        translated_text = response.choices[0].message.content.strip()
        update_progress("translation", 100, "完成", "翻譯完成！")
        return translated_text
        
    except Exception as e:
        error_str = str(e)
        print(f"❌ OpenAI 翻譯錯誤: {error_str}")
        
        # 處理常見的錯誤類型
        if "insufficient_quota" in error_str or "quota" in error_str.lower():
            raise Exception("翻譯失敗：API 配額不足。請檢查您的 OpenAI 帳戶餘額和計費設定。")
        elif "rate_limit" in error_str.lower():
            raise Exception("翻譯失敗：請求頻率過高，請稍後再試。")
        else:
            raise Exception(f"翻譯失敗: {error_str}")

def parse_srt_file(file_path: str) -> list:
    """解析 SRT 檔案"""
    print("📖 正在解析 SRT 檔案...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 按雙換行分割段落
    segments = []
    blocks = content.strip().split('\n\n')
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            # 第一行是序號，第二行是時間，第三行開始是文字
            index = lines[0]
            time_line = lines[1]
            text = '\n'.join(lines[2:])
            
            # 解析時間
            if ' --> ' in time_line:
                start_time, end_time = time_line.split(' --> ')
                segments.append({
                    'index': index,
                    'start_time': start_time,
                    'end_time': end_time,
                    'text': text
                })
    
    print(f"📊 解析完成，共 {len(segments)} 個字幕段落")
    return segments

def create_srt_file(segments: list, output_path: str):
    """生成 SRT 字幕檔案"""
    print("📝 [步驟 3/3] 開始生成 SRT 檔案...")
    
    try:
        print("📋 正在處理字幕段落...")
        print(f"📊 共 {len(segments)} 個字幕段落")
        
        # 寫入 SRT 檔案
        print("💾 正在寫入 SRT 檔案...")
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(segments, 1):
                start_time = format_time(segment['start'])
                end_time = format_time(segment['end'])
                text = segment['text'].strip()
                
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{text}\n\n")
        
        file_size = os.path.getsize(output_path)
        print(f"📁 SRT 檔案已生成: {output_path}")
        print(f"📊 檔案大小: {file_size:,} bytes")
        print("✅ [步驟 3/3] SRT 檔案生成完成！")
                    
    except Exception as e:
        print(f"❌ 生成 SRT 檔案失敗: {str(e)}")
        raise Exception(f"生成 SRT 檔案失敗: {str(e)}")

def create_txt_file(transcript: str, segments: list, output_path: str):
    """生成 TXT 文字檔案（英文內容，適合螢幕寬度的斷句）"""
    print("📝 [步驟 3/3] 開始生成 TXT 檔案...")
    
    try:
        # 收集所有文字並重新斷句
        print("📋 正在收集轉錄文字...")
        all_text = []
        for segment in segments:
            text = segment['text'].strip()
            if text:
                all_text.append(text)
        
        print(f"📊 收集到 {len(all_text)} 個文字段落")
        
        # 合併所有文字
        full_text = ' '.join(all_text)
        print(f"📏 總文字長度: {len(full_text)} 字元")
        
        # 重新斷句，適合螢幕寬度
        print("✂️ 正在進行智能斷句...")
        sentences = smart_sentence_split(full_text)
        print(f"📝 斷句完成: 共 {len(sentences)} 個句子")
        
        # 寫入檔案，每行一個句子
        print("💾 正在寫入 TXT 檔案...")
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, sentence in enumerate(sentences):
                f.write(sentence.strip())
                if i < len(sentences) - 1:  # 不是最後一個句子
                    f.write('\n')
                else:
                    f.write('\n')
        
        file_size = os.path.getsize(output_path)
        print(f"📁 TXT 檔案已生成: {output_path}")
        print(f"📊 檔案大小: {file_size:,} bytes")
        print("✅ [步驟 3/3] TXT 檔案生成完成！")
                    
    except Exception as e:
        print(f"❌ 生成 TXT 檔案失敗: {str(e)}")
        raise Exception(f"生成 TXT 檔案失敗: {str(e)}")

def smart_sentence_split(text: str) -> list:
    """智能斷句，限制在50個字元，遇到標點符號就斷句"""
    import re
    
    result = []
    current_sentence = ""
    
    # 按空格分割單詞
    words = text.split()
    
    for word in words:
        # 檢查加入這個詞後是否會超過50字元
        test_sentence = current_sentence + (" " if current_sentence else "") + word
        
        # 如果超過50字元，先保存當前句子
        if len(test_sentence) > 50 and current_sentence:
            result.append(current_sentence.strip())
            current_sentence = word
        else:
            current_sentence = test_sentence
        
        # 檢查是否遇到標點符號，如果遇到就斷句
        if word.endswith(('.', '!', '?', ':', ';')):
            result.append(current_sentence.strip())
            current_sentence = ""
        elif word.endswith(','):
            # 逗號後如果句子已經夠長（超過30字元），也可以斷句
            if len(current_sentence) > 30:
                result.append(current_sentence.strip())
                current_sentence = ""
    
    # 處理最後一個句子
    if current_sentence.strip():
        result.append(current_sentence.strip())
    
    # 確保不丟失任何內容，不過濾任何句子
    return result

def smart_sentence_split_chinese(text: str) -> list:
    """中文智能斷句，限制在25個字元，遇到標點符號就斷句"""
    import re
    
    result = []
    current_sentence = ""
    
    # 中文不需要按空格分割，直接按字符處理
    for char in text:
        test_sentence = current_sentence + char
        
        # 如果超過25字且當前句子不為空，就斷句
        if len(test_sentence) > 25 and current_sentence:
            result.append(current_sentence.strip())
            current_sentence = char
        else:
            current_sentence = test_sentence
        
        # 遇到標點符號就斷句
        if char in ('。', '！', '？', '：', '；', '.', '!', '?', ':', ';'):
            result.append(current_sentence.strip())
            current_sentence = ""
        elif char in ('，', ','):
            # 逗號後如果句子夠長就斷句
            if len(current_sentence) > 15:
                result.append(current_sentence.strip())
                current_sentence = ""
    
    if current_sentence.strip():
        result.append(current_sentence.strip())
    
    # 確保不丟失任何內容，不過濾任何句子
    return result

def format_time(seconds: float) -> str:
    """將秒數轉換為 SRT 時間格式"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace('.', ',')

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """首頁"""
    return templates.TemplateResponse("index.html", {"request": request})

# API 設定相關端點
@app.get("/api/check_api_status")
async def check_api_status():
    """檢查 API 狀態"""
    return {"connected": openai_client is not None}

@app.get("/api/progress")
async def get_progress():
    """獲取進度狀態"""
    return progress_data

@app.get("/api/get_api_key")
async def get_api_key():
    """取得儲存的 API 金鑰"""
    try:
        api_key = load_api_key()
        if api_key:
            # 只返回前8個字元和後4個字元，中間用星號代替
            masked_key = api_key[:8] + "*" * (len(api_key) - 12) + api_key[-4:] if len(api_key) > 12 else api_key
            return {"success": True, "api_key": masked_key}
        else:
            return {"success": False, "api_key": ""}
    except Exception as e:
        return {"success": False, "detail": str(e)}

@app.post("/api/test_api")
async def test_api(request: ApiKeyRequest):
    """測試 API 金鑰"""
    try:
        is_valid, message = test_openai_api(request.api_key)
        if is_valid:
            return {"success": True, "detail": message}
        else:
            return {"success": False, "detail": message}
    except Exception as e:
        return {"success": False, "detail": str(e)}

@app.post("/api/save_api_key")
async def save_api_key_endpoint(request: ApiKeyRequest):
    """儲存 API 金鑰"""
    try:
        # 先測試 API 金鑰是否有效
        is_valid, message = test_openai_api(request.api_key)
        if not is_valid:
            return {"success": False, "detail": f"API 金鑰無效: {message}"}
        
        # 儲存 API 金鑰
        if save_api_key(request.api_key):
            # 重新初始化 OpenAI 客戶端
            global openai_client
            openai_client = OpenAI(api_key=request.api_key)
            return {"success": True, "detail": "API 金鑰已儲存並生效"}
        else:
            return {"success": False, "detail": "儲存失敗"}
    except Exception as e:
        return {"success": False, "detail": str(e)}

@app.post("/generate_subtitle")
async def generate_subtitle(request: Request):
    """生成字幕的主要 API"""
    # 重置進度
    update_progress("transcription", 0, "準備中", "正在準備轉錄...")
    try:
        form = await request.form()
        youtube_url = form.get("youtube_url")
        model_selection = form.get("model_selection", "large-v3")
        filename = form.get("filename", "").strip()
        output_mode = form.get("output_mode", "txt")
        
        print(f"🎯 收到請求:")
        print(f"   📺 YouTube URL: {youtube_url}")
        print(f"   🤖 模型選擇: {model_selection}")
        print(f"   📝 自訂檔名: {filename if filename else '自動生成'}")
        print(f"   📄 輸出模式: {output_mode}")
        
        if not youtube_url:
            raise HTTPException(status_code=400, detail="請提供 YouTube URL")
        
        if not validate_youtube_url(youtube_url):
            raise HTTPException(status_code=400, detail="無效的 YouTube URL")
        
        if model_selection not in ["base", "large-v3"]:
            raise HTTPException(status_code=400, detail="無效的模型選擇")
        
        if output_mode not in ["txt", "srt"]:
            raise HTTPException(status_code=400, detail="無效的輸出模式")
        
        # 建立臨時目錄
        temp_dir = tempfile.mkdtemp(dir="temp")
        unique_id = str(uuid.uuid4())
        
        try:
            # 下載音訊
            audio_path = download_audio(youtube_url, temp_dir)
            
            # 轉錄音訊
            print(f"開始轉錄音訊: {audio_path}")
            result = transcribe_audio(audio_path, model_selection)
            print(f"轉錄完成，共 {len(result['segments'])} 個段落")
            
            # 生成檔案
            if output_mode == "txt":
                # 生成 TXT 檔案
                if filename:
                    output_filename = f"{filename}.txt"
                else:
                    output_filename = f"transcript_{unique_id}.txt"
                
                output_path = os.path.join("temp", output_filename)
                create_txt_file(result["text"], result["segments"], output_path)
                print(f"TXT 檔案已生成: {output_path}")
                
                return {
                    "success": True,
                    "message": "文字轉錄成功！",
                    "download_url": f"/download/{output_filename}",
                    "filename": output_filename
                }
            else:
                # 生成 SRT 檔案
                if filename:
                    output_filename = f"{filename}.srt"
                else:
                    output_filename = f"subtitle_{unique_id}.srt"
                
                output_path = os.path.join("temp", output_filename)
                create_srt_file(result["segments"], output_path)
                print(f"SRT 檔案已生成: {output_path}")
                
                return {
                    "success": True,
                    "message": "字幕生成成功！",
                    "download_url": f"/download/{output_filename}",
                    "filename": output_filename
                }
            
        except Exception as e:
            # 清理臨時檔案
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise HTTPException(status_code=500, detail=str(e))
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"伺服器錯誤: {str(e)}")

@app.post("/translate_file")
async def translate_file(
    file: UploadFile = File(...),
    target_language: str = "zh-TW",
    translation_filename: str = "",
    translation_output_mode: str = "txt"
):
    """翻譯檔案的主要 API"""
    # 重置進度
    update_progress("translation", 0, "準備中", "正在準備翻譯...")
    try:
        print(f"🎯 收到翻譯請求:")
        print(f"   📁 檔案名稱: {file.filename}")
        print(f"   🌍 目標語言: {target_language}")
        print(f"   📝 自訂檔名: {translation_filename if translation_filename else '自動生成'}")
        print(f"   📄 輸出模式: {translation_output_mode}")
        
        if not openai_client:
            raise HTTPException(status_code=500, detail="OpenAI API 未配置，請檢查 API 金鑰")
        
        # 檢查檔案格式
        if not file.filename.endswith(('.txt', '.srt')):
            raise HTTPException(status_code=400, detail="只支援 TXT 和 SRT 格式的檔案")
        
        # 儲存上傳的檔案
        unique_id = str(uuid.uuid4())
        temp_file_path = os.path.join("temp", f"upload_{unique_id}_{file.filename}")
        
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        print(f"📁 檔案已儲存: {temp_file_path}")
        
        try:
            # 根據檔案格式處理
            if file.filename.endswith('.srt'):
                # 處理 SRT 檔案
                segments = parse_srt_file(temp_file_path)
                
                # 翻譯每個段落
                print("🔄 開始翻譯字幕段落...")
                translated_segments = []
                for i, segment in enumerate(segments):
                    print(f"📝 翻譯段落 {i+1}/{len(segments)}")
                    translated_text = translate_text_with_openai(segment['text'], target_language)
                    translated_segments.append({
                        'index': segment['index'],
                        'start_time': segment['start_time'],
                        'end_time': segment['end_time'],
                        'text': translated_text
                    })
                
                # 生成輸出檔案
                if translation_output_mode == "srt":
                    # 生成翻譯後的 SRT 檔案
                    if translation_filename:
                        output_filename = f"{translation_filename}.srt"
                    else:
                        output_filename = f"translated_{unique_id}.srt"
                    
                    output_path = os.path.join("temp", output_filename)
                    create_translated_srt_file(translated_segments, output_path)
                    
                    return {
                        "success": True,
                        "message": "字幕翻譯成功！",
                        "download_url": f"/download/{output_filename}",
                        "filename": output_filename
                    }
                else:
                    # 生成 TXT 檔案
                    if translation_filename:
                        output_filename = f"{translation_filename}.txt"
                    else:
                        output_filename = f"translated_{unique_id}.txt"
                    
                    output_path = os.path.join("temp", output_filename)
                    create_translated_txt_file(translated_segments, output_path)
                    
                    return {
                        "success": True,
                        "message": "文字翻譯成功！",
                        "download_url": f"/download/{output_filename}",
                        "filename": output_filename
                    }
            else:
                # 處理 TXT 檔案
                print("📖 正在讀取 TXT 檔案...")
                with open(temp_file_path, 'r', encoding='utf-8') as f:
                    text_content = f.read()
                
                print(f"📊 檔案內容長度: {len(text_content)} 字元")
                
                # 翻譯文字
                translated_text = translate_text_with_openai(text_content, target_language)
                
                # 生成輸出檔案
                if translation_filename:
                    output_filename = f"{translation_filename}.txt"
                else:
                    output_filename = f"translated_{unique_id}.txt"
                
                output_path = os.path.join("temp", output_filename)
                
                # 使用中文斷句功能生成 TXT 檔案
                create_translated_txt_file([{'text': translated_text}], output_path)
                
                return {
                    "success": True,
                    "message": "文字翻譯成功！",
                    "download_url": f"/download/{output_filename}",
                    "filename": output_filename
                }
        
        finally:
            # 清理上傳的臨時檔案
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                print(f"🗑️ 已清理臨時檔案: {temp_file_path}")
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ 翻譯錯誤: {str(e)}")
        raise HTTPException(status_code=500, detail=f"翻譯失敗: {str(e)}")

def create_translated_srt_file(segments: list, output_path: str):
    """生成翻譯後的 SRT 檔案"""
    print("📝 正在生成翻譯後的 SRT 檔案...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for segment in segments:
            f.write(f"{segment['index']}\n")
            f.write(f"{segment['start_time']} --> {segment['end_time']}\n")
            f.write(f"{segment['text']}\n\n")
    
    print(f"✅ 翻譯後的 SRT 檔案已生成: {output_path}")

def create_translated_txt_file(segments: list, output_path: str):
    """生成翻譯後的 TXT 檔案（中文內容，適合螢幕寬度的斷句）"""
    print("📝 正在生成翻譯後的 TXT 檔案...")
    
    # 將所有翻譯後的文字合併
    full_text = ""
    for segment in segments:
        if segment.get('text'):
            full_text += segment['text'] + " "
    
    # 清理文字
    full_text = full_text.strip()
    
    # 對中文進行智能斷句
    print("🔤 正在進行中文智能斷句...")
    sentences = smart_sentence_split_chinese(full_text)
    
    # 寫入檔案
    with open(output_path, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            f.write(sentence + "\n\n")
    
    print(f"✅ 翻譯後的 TXT 檔案已生成: {output_path}")
    print(f"📊 共生成 {len(sentences)} 個段落")

@app.get("/download/{filename}")
async def download_file(filename: str):
    """下載檔案 (TXT 或 SRT)"""
    file_path = os.path.join("temp", filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="檔案不存在")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='application/octet-stream'
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)