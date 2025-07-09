from flask_cors import CORS
from flask import Flask, request, jsonify
import whisper
import os
import spacy
import uuid
import requests  # 用于与 PHP 后端通信
from pydub import AudioSegment
from flask_sqlalchemy import SQLAlchemy
import langdetect
from datetime import datetime, timedelta
import dateparser
from dateparser.search import search_dates
import re
import pytz


nlp_zh = spacy.load("C:/Users/User/voice-memo-assistant/zh_text_categorizer_model")
nlp_en = spacy.load("C:/Users/User/voice-memo-assistant/en_text_categorizer_model")

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///memo.db'
db = SQLAlchemy(app)
CORS(app)

# Load Whisper model
model = whisper.load_model("base")

class Memo(db.Model):
    __tablename__ = 'memo'
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String(500), nullable=False)
    category = db.Column(db.String(100), nullable=False)
    timestamp = db.Column(db.String(100), nullable=False)
    userID = db.Column(db.Integer, nullable=True)

with app.app_context():
    db.create_all()

UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# API 地址为 PHP 后端地址，您可以根据需要修改
PHP_API_URL = 'http://192.168.0.17:8000/login.php'

# 模拟登录功能（通过 PHP 后端验证）
@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")
    print("Login attempt with:", username, password)

    try:
        # 发送请求到 PHP 后端验证用户登录
        response = requests.post(PHP_API_URL, data={'username': username, 'password': password})

        # 如果请求成功，处理响应
        response.raise_for_status()  # 确保请求成功（状态码为 200）

        # 获取 PHP 后端的响应
        response_data = response.json()

        if response_data.get("success"):
            return jsonify({"success": True})
        else:
            return jsonify({"success": False, "message": response_data.get("message", "Invalid credentials")}), 401

    except requests.exceptions.RequestException as e:
        # 捕获请求异常
        print("Login request error:", e)
        return jsonify({"success": False, "message": "Server error"}), 500
    except Exception as e:
        # 捕获其他异常
        print("Login error:", e)
        return jsonify({"success": False, "message": "Server error"}), 500

def detect_language(text):
    try:
        lang = langdetect.detect(text)
        return lang  # 返回语言代码，如 'en' 或 'zh'
    except langdetect.lang_detect_exception.LangDetectException:
        return None  # 如果无法检测到语言，返回 None

def categorize_text(text, threshold=0.5):
    language = detect_language(text)

    if language == 'zh':
        doc = nlp_zh(text)
    elif language == 'en':
        doc = nlp_en(text)
    else:
        return "Others"

    if doc and doc.cats:
        print("Classification probabilities:", doc.cats)
        category, prob = max(doc.cats.items(), key=lambda item: item[1])
        print(f"Category: {category}, Probability: {prob}")
        if category in ["Study", "Work", "Daily"] and prob >= threshold:
            return category
        else:
            return "Others"  # 分类概率低或者类别不在预定义范围内
    return "Others"  # 如果没有符合条件的分类，则返回 Unknown

# 你已有的 predict_and_extract_time() 函数
def predict_and_extract_time(text):
    language = detect_language(text)
    if language == 'zh':
        doc = nlp_zh(text)
    elif language == 'en':
        doc = nlp_en(text)
    else:
        # 无法判断语言，默认使用中文模型或返回 Others
        doc = nlp_zh(text)

    if doc and doc.cats:
        category = max(doc.cats, key=doc.cats.get)
    else:
        category = "Others"

    time = parse_time(text)
    return category, time

def extract_task_title(text):
    import re

    # 1. 清除日期表达式（中英文格式）
    text = re.sub(r"\d{1,2}(st|nd|rd|th)? of [A-Za-z]+", "", text)  # 英文 23rd of June
    text = re.sub(r"\d{4}-\d{1,2}-\d{1,2}", "", text)               # 2025-06-23
    text = re.sub(r"\d+月\d+(日|号)?", "", text)                     # 6月23日 / 6月23号

    # 2. 去除中英文模糊时间词（今天、明天、下午等）
    time_words = [
        "今天", "明天", "后天", "早上", "下午", "晚上", "上午",
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
        "morning", "afternoon", "evening", "tonight", "tomorrow", "today"
    ]
    for word in time_words:
        text = text.replace(word, "")

    # 3. 去除常见前缀指令（中英文删除/取消表达）
    prefix_patterns = [
        r"^(请)?(帮我)?(把)?(我要)?(删除|取消|移除)",          # 中文
        r"^(please )?(help me )?(delete|remove|cancel|erase|clear)( the memo of| the)?",  # 英文
    ]
    for pattern in prefix_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # 4. 去除结尾修饰词（中英文“的任务”、“的事情”、“的行程”...）
    suffix_patterns = [
        r"(的)?(任务|事情|安排|行程|memo|note|schedule|event)?(删掉|删除|取消)?$",
    ]
    for pattern in suffix_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # 5. 去除多余空格
    return text.strip()


def query_tasks_by_date(date_str):
    # 假设你有 Memo 表，timestamp字段是 ISO格式字符串
    tasks = Memo.query.filter(Memo.timestamp.like(f"{date_str}%")).all()
    return [{"text": t.text, "category": t.category, "timestamp": t.timestamp} for t in tasks]

# 解析中文或英文时间表达
def parse_time(text, only_date=False):
    now = datetime.now()
    try:
        text_cleaned = text.replace("號", "号").replace("日", "号")

        # 优先判断有没有用户明确说出“几月几日”或“几号”
        zh_date_match = re.search(r"(\d{1,2})月(\d{1,2})[号日]?", text_cleaned)
        if zh_date_match:
            print("✅ 明确中文日期:", zh_date_match.group(0))
            month = int(zh_date_match.group(1))
            day = int(zh_date_match.group(2))
            year = now.year
            parsed = datetime(year, month, day)

            # 🚫 删除类任务时不要跳年
            allow_next_year = not (
                "delete" in text.lower() or "刪除" in text or "删除" in text
            )

            if parsed < now and allow_next_year:
                parsed = parsed.replace(year=year + 1)

            return parsed.strftime("%Y-%m-%d") if only_date else parsed.strftime("%Y-%m-%d %H:%M:%S")

        # 手动检测英文日期格式 like "23 June" or "June 23"
        en_date_match = re.search(
            r'(?:(\d{1,2})(?:st|nd|rd|th)?\s+(January|February|March|April|May|June|July|August|September|October|November|December)|'  # 24 June
            r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:st|nd|rd|th)?)',
            text,
            re.IGNORECASE
        )
        if en_date_match:
            print("📝 手动匹配英文日期:", en_date_match.group(0))
            if en_date_match.group(1) and en_date_match.group(2):  # case: 24 June
                day = int(en_date_match.group(1))
                month_str = en_date_match.group(2)
            elif en_date_match.group(3) and en_date_match.group(4):  # case: June 24
                day = int(en_date_match.group(4))
                month_str = en_date_match.group(3)
            else:
                return None

            month = datetime.strptime(month_str[:3], "%b").month
            year = now.year
            parsed = datetime(year, month, day)

            is_delete = any(word in text.lower() for word in ["delete", "remove", "清除", "刪除"])
            if parsed < now and not is_delete:
                parsed = parsed.replace(year=year + 1)

            return parsed.strftime("%Y-%m-%d") if only_date else parsed.strftime("%Y-%m-%d %H:%M:%S")


        # ✅ 使用 search_dates 解析
        parsed_result = search_dates(text_cleaned, languages=['zh', 'en'], settings={
            'PREFER_DATES_FROM': 'future',
            'RELATIVE_BASE': now
        })
        print("🧠 search_dates解析:", parsed_result)

        if parsed_result:
            matched_text, dt = parsed_result[0]

            # ✅ 新增：识别“几点”并手动修正时间
            hour_match = re.search(r"(晚上|下午)?(\d{1,2})点", text)
            if hour_match:
                hour = int(hour_match.group(2))
                if hour < 12 and hour_match.group(1) in ["晚上", "下午"]:
                    hour += 12
                dt = dt.replace(hour=hour, minute=0, second=0, microsecond=0)

            is_delete = any(word in text.lower() for word in ["delete", "刪除", "删除"])
            # 如果没明确日期，就用今天 +1 推理；否则用原解析时间
            has_explicit_date = bool(re.search(
                r"(\d{1,2})月(\d{1,2})[号日]?|today|tomorrow|[0-9]{1,2} [A-Za-z]+|明天|后天",
                matched_text.lower()
            ))
            if not has_explicit_date:
                dt = dt.replace(year=now.year, month=now.month, day=now.day)
                if dt < now and "今天" not in text and "today" not in text.lower():
                    dt += timedelta(days=1)
            elif not is_delete and dt < now:
                dt = dt.replace(year=now.year + 1)

            return dt.strftime("%Y-%m-%d") if only_date else dt.strftime("%Y-%m-%d %H:%M:%S")

        # 🔁 fallback：dateparser 处理马来文或其他
        fallback_dt = dateparser.parse(text_cleaned, settings={
            'PREFER_DATES_FROM': 'future',
            'RELATIVE_BASE': now
        })
        if fallback_dt:
            print("📦 fallback dateparser解析:", fallback_dt)
            return fallback_dt.strftime("%Y-%m-%d") if only_date else fallback_dt.strftime("%Y-%m-%d %H:%M:%S")

    except Exception as e:
        print("❌ parse_time error:", str(e))

    return None

def to_simplified(text):
    replacements = {
        "讀": "读",
        "學": "学",
        "寫": "写",
        "說": "说",
        "會議": "会议",
        "備忘錄": "备忘录",
        "開會": "开会",
        "聽": "听",
        "訊息": "信息",
        "電腦": "电脑",
        "報告": "报告",
        "設計": "设计",
        "書": "书",
        "請": "请",
        "刪除": "删除",
        "取消": "取消",
        "安排": "安排",
        "事情": "事情",
        "行程": "行程",
        "活動": "活动",
        "聯絡": "联络",
        "討論": "讨论",
        "報名": "报名",
        "練習": "练习",
        "報到": "报到",
        "系統": "系统",
        "頁面": "页面",
        "功能": "功能",
        "分類": "分类",
        "訊號": "信号",
        "開始": "开始",
        "結束": "结束",
        "備份": "备份",
        "簡報": "简报",
        "提醒": "提醒",
        "紀錄": "记录",
        "備註": "备注",
        "檔案": "文件",
        "選項": "选项",
        "設定": "设置",
        "日曆": "日历",
        "標題": "标题",
        "檢查": "检查",
        "號": "号",
        "報": "报",
        "點": "点",
        "郵件": "邮件",
        "客戶": "客户",
        "任務": "任务"
        # 👉 你可以继续加常用词
    }
    for traditional, simplified in replacements.items():
        text = text.replace(traditional, simplified)
    return text
# 音频转录功能
@app.route("/transcribe", methods=["POST"])
def transcribe():
    user_id = request.form.get('user_id')  # 从 FormData 获取
    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400

    print("Received request:", request)
    if "file" not in request.files:
        print("No file part")
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    print("Received file:", file.filename)

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        # 保存上传音频
        original_ext = file.filename.split('.')[-1]
        original_filename = f"{uuid.uuid4()}.{original_ext}"
        original_path = os.path.join(UPLOAD_FOLDER, original_filename)
        file.save(original_path)
        print(f"Uploaded file saved as: {original_path}")

        # 转为 WAV 格式
        if original_ext.lower() != 'wav':
            audio = AudioSegment.from_file(original_path)
            wav_filename = f"{uuid.uuid4()}.wav"
            wav_path = os.path.join(UPLOAD_FOLDER, wav_filename)
            audio.export(wav_path, format="wav")
            print(f"Converted to WAV: {wav_path}")
        else:
            wav_path = original_path

        # 使用 Whisper 识别语音
        result = model.transcribe(wav_path)
        transcription = result["text"].strip()
        print("Transcription result:", transcription)
        # ✅ 繁体转简体（一定要在分类模型之前做）
        transcription = to_simplified(transcription)
        print("✅ Simplified transcription:", transcription)


        # NLP模型分类（自动检测语种）
        if re.search(r"[a-zA-Z]", transcription):  # 含有英文字母
            doc = nlp_en(transcription)
            print("🧠 使用英文模型")
        else:
            doc = nlp_zh(transcription)
            print("🧠 使用中文模型")

        print("Text category probabilities:", doc.cats)
        category, prob = max(doc.cats.items(), key=lambda item: item[1])
        print(f"Predicted category: {category}, Probability: {prob}")
        # 删除类指令处理
        if category == "Delete_Specific" or category == "Delete_All":
            memo_time = parse_time(transcription)
            print(f"🗑️ Delete request | Type: {category} | Time: {memo_time}")
            print(f"[DEBUG] parse_time output: {memo_time}")

            keyword = extract_task_title(transcription) if category == "Delete_Specific" else None

            if memo_time:
                try:
                    date_obj = datetime.strptime(memo_time, "%Y-%m-%d %H:%M:%S")
                except:
                    try:
                        date_obj = datetime.strptime(memo_time, "%Y-%m-%d")
                    except:
                        date_obj = None
            else:
                date_obj = None

            if date_obj:
                start_datetime = date_obj.replace(hour=0, minute=0, second=0)
                end_datetime = date_obj.replace(hour=23, minute=59, second=59)
                keyword_for_php = None  # ✅ 如果有明确时间 ➜ 不传 keyword
            else:
                print("⚠️ 未识别出日期 ➜ 将使用关键词删除（全时间范围）")
                start_datetime = datetime(2000, 1, 1)
                end_datetime = datetime(2100, 12, 31)
                keyword_for_php = keyword or transcription  # ✅ fallback 使用原句

            print(f"🗑️ Delete time range: {start_datetime} to {end_datetime}")
            print(f"🗑️ Keyword for PHP: {keyword_for_php}")

            return jsonify({
                "transcription": transcription,
                "category": category,
                "category_id": 0,
                "is_query": False,
                "need_confirm": True,
                "pending_delete": {
                    "start_time": start_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                    "end_time": end_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                    "keyword": keyword_for_php,
                    "category": category
                }
            })

        # 查询类问题处理（Query_Today、Query_Tomorrow、Query_Custom）
        if category.startswith("Query_"):
            def get_query_date(cat, text):
                if cat == "Query_Today":
                    return datetime.now().strftime("%Y-%m-%d")
                elif cat == "Query_Tomorrow":
                    return (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
                elif cat == "Query_Custom":
                    parsed_time = parse_time(text)
                    print(f"Custom query date from text: {parsed_time}")
                    return parsed_time if parsed_time else datetime.now().strftime("%Y-%m-%d")
                else:
                    return datetime.now().strftime("%Y-%m-%d")
            print("Original query:", transcription)
            query_date = get_query_date(category, transcription)
            print("Parsed query date:", query_date)
            print(f"Query date resolved as: {query_date}")

            # 查询数据库任务
            try:
                query_date_dt = datetime.strptime(query_date, "%Y-%m-%d %H:%M:%S")
            except:
                query_date_dt = datetime.now()

            start_datetime = query_date_dt.replace(hour=0, minute=0, second=0)
            end_datetime = query_date_dt.replace(hour=23, minute=59, second=59)

            tasks = Memo.query.filter(
                Memo.userID == user_id,
                Memo.timestamp >= start_datetime,
                Memo.timestamp <= end_datetime
            ).all()
            print("Querying tasks between", start_datetime, "and", end_datetime)
            print("Tasks found:", tasks)
            tasks_data = [
                {"text": t.text, "category": t.category, "timestamp": t.timestamp}
                for t in tasks
            ]

            return jsonify({
                "transcription": transcription,
                "category": category,
                "category_id": 0,
                "is_query": True,
                "need_confirm": True,
                "query_date": query_date,
                "tasks": tasks_data
            })

        # 普通备忘录分类
        category_map = {
            "Study": 1,
            "Work": 2,
            "Daily": 3
        }
        category_id = category_map.get(category, 4)  # Others 为 4
        memo_time = parse_time(transcription)

        # 统一格式化时间（防止前端接收到中文“上午1:35:54”格式）
        if memo_time:
            try:
                dt = datetime.strptime(memo_time, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                try:
                    dt = datetime.strptime(memo_time, "%Y-%m-%d")
                except:
                    dt = datetime.now()
        else:
            dt = datetime.now()

        formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
        iso_time = dt.isoformat()

        return jsonify({
            "transcription": transcription,
            "category": category,
            "category_id": category_id,
            "is_query": False,
            "time": memo_time,
            "iso_time": iso_time
        })

    except Exception as e:
        print("Transcription failed:", str(e))
        return jsonify({"error": "Transcription failed."}), 500


# 这是修改后的 classify_text 函数
def classify_text(text, threshold=0.5):
    language = detect_language(text)

    if language == 'zh':
        doc = nlp_zh(text)
    elif language == 'en':
        doc = nlp_en(text)
    else:
        return {"category": "Others", "category_id": 4}

    category_map = {
        "Study": 1,
        "Work": 2,
        "Daily": 3,
        "Others": 4
    }
    reverse_map = {v: k for k, v in category_map.items()}

    if doc and doc.cats:
        print("Text category probabilities:", doc.cats)
        category, prob = max(doc.cats.items(), key=lambda item: item[1])
        print(f"Predicted category: {category}, Probability: {prob}")

        # 如果预测类别不在 map 中，则归为 Others
        if category not in category_map:
            category = "Others"

        # 如果预测类别概率低于阈值，也归为 Others
        if prob < threshold:
            category = "Others"

        return {
            "category": category,
            "category_id": category_map.get(category, 4)
        }

    return {"category": "Others", "category_id": 4}


# 修改后的 classify 路由
@app.route('/classify', methods=['POST'])
def classify():
    text = request.json.get('text', '')
    print(f"Received text: {text}")  # 打印收到的文本
    result = classify_text(text)
    print(f"Classify result: {result}")  # 打印分类结果
    return jsonify(result)  # 返回 JSON 格式的结果


@app.route("/save_memo", methods=["POST"])
def save_memo():
    try:
        data = request.get_json()
        user_id = data.get('userID')
        title = data.get('title')
        category_id = data.get('category_id')
        time = data.get('time') or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if not user_id or not title or not category_id:
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400

        new_memo = Memo(
            text=title,
            category=str(category_id),
            timestamp=time,
            userID=user_id
        )
        db.session.add(new_memo)
        db.session.commit()

        return jsonify({'success': True, 'message': 'Memo saved successfully'}), 200

    except Exception as e:
        print("Error saving memo:", str(e))
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route("/save_and_list_memos", methods=["POST"])
def save_and_list_memos():
    try:
        data = request.get_json()
        user_id = data.get('userID')
        title = data.get('title')
        category_id = data.get('category_id')
        time = data.get('time') or datetime.now().strftime("%Y-%m-%d")

        if not user_id or not title or not category_id:
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400

        new_memo = Memo(
            userID=user_id,
            text=title,
            category=str(category_id),
            timestamp=time
        )
        db.session.add(new_memo)
        db.session.commit()

        # 查询当前用户所有备忘录
        memos = Memo.query.filter_by(userID=user_id).order_by(Memo.timestamp.desc()).all()
        memo_list = [{
            "id": m.id,
            "text": m.text,
            "category": m.category,
            "timestamp": m.timestamp
        } for m in memos]

        return jsonify({'success': True, 'memos': memo_list}), 200

    except Exception as e:
        print("Error saving/listing memos:", str(e))
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
