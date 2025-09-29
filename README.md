# 🎬 YouTube 文字轉錄器 & 翻譯工具

一個功能強大的 Web 應用程式，可以將 YouTube 影片轉錄成文字，並支援多語言翻譯。

## ✨ 功能特色

- 🎵 **YouTube 轉錄**：支援任何 YouTube 影片的音訊轉錄
- 🌍 **多語言翻譯**：支援繁體中文、簡體中文、英文、日文、韓文等
- 📝 **智能斷句**：自動優化文字格式，適合閱讀
- 📄 **多種輸出格式**：支援 TXT 和 SRT 格式
- 🎯 **進度顯示**：實時顯示處理進度
- ⚙️ **API 設定**：網頁內直接設定 OpenAI API

## 🚀 快速開始

### 本地運行

1. 安裝依賴
```bash
pip install -r requirements.txt
```

2. 設定環境變數
```bash
# 創建 .env 檔案
OPENAI_API_KEY=your_api_key_here
```

3. 運行應用
```bash
python main.py
```

4. 打開瀏覽器
```
http://localhost:8000
```

### 線上部署

#### 使用 Render（推薦）

1. Fork 這個倉庫
2. 註冊 [Render.com](https://render.com)
3. 創建新的 Web Service
4. 連接 GitHub 倉庫
5. 設定環境變數：`OPENAI_API_KEY`
6. 部署完成！

## 📋 系統需求

- Python 3.11+
- FFmpeg（用於音訊處理）
- OpenAI API 金鑰

## 🔧 技術架構

- **後端**：FastAPI
- **前端**：HTML/CSS/JavaScript
- **AI 模型**：OpenAI Whisper
- **翻譯**：OpenAI GPT-3.5-turbo

## 📝 使用說明

### YouTube 轉錄
1. 貼上 YouTube 連結
2. 選擇 Whisper 模型（base 或 large-v3）
3. 設定輸出檔名和格式
4. 點擊「開始轉錄」

### 檔案翻譯
1. 上傳 TXT 或 SRT 檔案
2. 選擇目標語言
3. 設定輸出檔名和格式
4. 點擊「開始翻譯」

## ⚙️ API 設定

在網頁右上角點擊「⚙️ API 設定」：
1. 輸入 OpenAI API 金鑰
2. 點擊「🧪 測試 API」驗證
3. 點擊「💾 儲存設定」

## 🛠️ 故障排除

### 常見問題

1. **API 配額不足**
   - 檢查 OpenAI 帳戶餘額
   - 設定計費方式

2. **轉錄失敗**
   - 確認 YouTube 連結有效
   - 檢查網路連線

3. **檔案上傳失敗**
   - 確認檔案格式（TXT 或 SRT）
   - 檢查檔案大小限制

## 📄 授權

MIT License

## 🤝 貢獻

歡迎提交 Issue 和 Pull Request！

## 📞 支援

如有問題，請在 GitHub 上提交 Issue。