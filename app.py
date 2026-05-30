import os, re, json, math, difflib, statistics
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge
import anthropic
import mammoth
import requests as req_lib

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024

@app.errorhandler(RequestEntityTooLarge)
def too_large(e):
    return jsonify({"error": "File quá lớn! Vui lòng dùng file dưới 200MB."}), 413

os.makedirs('uploads', exist_ok=True)

# ─── Text Extraction ──────────────────────────────────────────────────────────

def extract_text_from_file(file_storage):
    filename = file_storage.filename.lower()
    content = file_storage.read()
    if filename.endswith('.txt'):
        return content.decode('utf-8', errors='ignore')
    elif filename.endswith('.docx'):
        import io
        return mammoth.extract_raw_text(io.BytesIO(content)).value
    elif filename.endswith('.doc'):
        import io, tempfile
        try:
            r = mammoth.extract_raw_text(io.BytesIO(content))
            if r.value.strip(): return r.value
        except: pass
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.doc', delete=False) as tmp:
                tmp.write(content); tmp_path = tmp.name
            with open(tmp_path, 'rb') as f:
                return mammoth.extract_raw_text(f).value
        except:
            return content.decode('utf-8', errors='ignore')
        finally:
            if tmp_path:
                try: os.unlink(tmp_path)
                except: pass
    return content.decode('utf-8', errors='ignore')

# ─── Helpers ──────────────────────────────────────────────────────────────────

def split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?।\n])\s+', text) if len(s.strip()) > 10]

def remove_template_content(main_text, template_text, threshold=0.75):
    tmpl_sents = set(s.lower().strip() for s in split_sentences(template_text) if len(s) > 15)
    main_sents = split_sentences(main_text)
    filtered, excluded = [], 0
    for s in main_sents:
        sl = s.lower().strip()
        hit = any(difflib.SequenceMatcher(None, sl, t).ratio() >= threshold for t in tmpl_sents)
        if hit: excluded += 1
        elif len(s.strip()) > 10: filtered.append(s)
    return ' '.join(filtered), excluded / max(len(main_sents), 1)

def chunk_text(text, chunk_size=1500):
    sents = split_sentences(text)
    if not sents: return [text] if len(text) > 30 else []
    chunks, cur, cur_len = [], [], 0
    for s in sents:
        cur.append(s); cur_len += len(s)
        if cur_len >= chunk_size:
            chunks.append(' '.join(cur))
            cur = cur[-2:] if len(cur) > 2 else []
            cur_len = sum(len(x) for x in cur)
    if cur: chunks.append(' '.join(cur))
    return [c for c in chunks if len(c) > 50]

def chunk_text_no_overlap(text, chunk_size=1500):
    """Chunk KHÔNG overlap — dùng cho document view để map chính xác."""
    sents = split_sentences(text)
    if not sents: return [text] if len(text) > 30 else []
    chunks, cur, cur_len = [], [], 0
    for s in sents:
        cur.append(s); cur_len += len(s)
        if cur_len >= chunk_size:
            chunks.append(' '.join(cur))
            cur = []; cur_len = 0
    if cur: chunks.append(' '.join(cur))
    return [c for c in chunks if len(c) > 50]

# ─── AI Detection Engine (nâng cấp mạnh hơn) ─────────────────────────────────

# Danh sách mở rộng các đặc trưng ChatGPT/AI tiếng Việt và tiếng Anh
CHATGPT_VI_PATTERNS = [
    # Mở đầu đoạn kiểu AI
    r'\btất nhiên\b', r'\bchắc chắn rằng\b', r'\bkhông thể phủ nhận\b',
    r'\btrong bối cảnh\b', r'\btrong thời đại\b', r'\bngày nay\b',
    r'\bnhư chúng ta (đã )?biết\b', r'\bcó thể (thấy|nói|khẳng định) rằng\b',
    r'\bđiều này cho thấy\b', r'\bđiều đó có nghĩa\b',
    r'\bvấn đề (này|đó|trên) (là|cần|đòi hỏi)\b',
    # Kết luận kiểu AI
    r'\btóm lại\b', r'\bnhìn chung\b', r'\bkết luận lại\b',
    r'\bqua (những|các) (phân tích|nghiên cứu|ví dụ) trên\b',
    r'\btừ (những|các) (điều|vấn đề|phân tích) trên\b',
    # Chuyển đoạn kiểu AI
    r'\bbên cạnh đó\b', r'\bngoài ra\b', r'\bhơn nữa\b', r'\bđồng thời\b',
    r'\bthêm vào đó\b', r'\bbên cạnh những (điều|ưu điểm|hạn chế)\b',
    r'\btrong khi đó\b', r'\bngược lại\b', r'\btuy nhiên\b.*\bnhưng\b',
    # Nhấn mạnh AI
    r'\bquan trọng (là|hơn|nhất)\b', r'\bcần lưu ý (rằng|là)\b',
    r'\bđáng chú ý (là|rằng)\b', r'\bđặc biệt (là|quan trọng)\b',
    r'\bđây là (một|điều|vấn đề)\b',
    # Liệt kê AI
    r'\bthứ (nhất|hai|ba|tư|năm)\b.*\bthứ (hai|ba|tư|năm)\b',
    r'\b(đầu tiên|tiếp theo|cuối cùng)\b',
    r'\bmột (là|mặt)\b.*\bhai (là|mặt)\b',
]

CHATGPT_EN_PATTERNS = [
    r'\bfirstly\b.*\bsecondly\b', r'\bin conclusion\b', r'\bin summary\b',
    r'\bfurthermore\b', r'\bmoreover\b', r'\badditionally\b', r'\bnotably\b',
    r'\bit is worth noting\b', r'\bit is important to\b', r'\bsignificantly\b',
    r'\bthis highlights\b', r'\bthis demonstrates\b', r'\bthis suggests\b',
    r'\bin other words\b', r'\bto summarize\b', r'\boverall\b',
    r'\bas a result\b', r'\bconsequently\b', r'\btherefore\b.*\bit\b',
]

# Từ "siêu hàn lâm" mà AI hay dùng nhưng người thường ít dùng
ACADEMIC_OVERUSE = [
    'toàn diện', 'hiệu quả', 'tích cực', 'bền vững', 'phát triển',
    'nâng cao', 'đổi mới', 'triển khai', 'thực hiện', 'đảm bảo',
    'chất lượng', 'năng lực', 'hệ thống', 'quá trình', 'phương pháp',
    'giải pháp', 'chiến lược', 'mục tiêu', 'kết quả', 'đánh giá',
    'comprehensive', 'effective', 'significant', 'important', 'various',
    'essential', 'crucial', 'innovative', 'sustainable', 'optimize',
]

def heuristic_ai_score(text):
    score = 0
    reasons = []
    lower = text.lower()
    sents = [s.strip() for s in re.split(r'[.!?।]+', text) if len(s.strip()) > 8]
    if not sents:
        return 0, []

    # ── 1. Burstiness: AI câu đều, người thật câu lộn xộn ──────────────────
    lengths = [len(s.split()) for s in sents]
    if len(lengths) >= 3:
        try:
            stdev = statistics.stdev(lengths)
            mean = statistics.mean(lengths)
            cv = stdev / mean if mean > 0 else 1  # coefficient of variation
            if cv < 0.25:
                score += 30
                reasons.append(f"Câu rất đều nhau (CV={cv:.2f}) — dấu hiệu rõ của AI")
            elif cv < 0.40:
                score += 15
                reasons.append(f"Câu khá đồng đều (CV={cv:.2f})")
        except: pass

    # ── 2. Đếm pattern AI tiếng Việt ────────────────────────────────────────
    vi_hits = sum(1 for p in CHATGPT_VI_PATTERNS if re.search(p, lower))
    if vi_hits >= 6:
        score += 35
        reasons.append(f"Rất nhiều cấu trúc câu AI tiếng Việt ({vi_hits} pattern)")
    elif vi_hits >= 3:
        score += 22
        reasons.append(f"Nhiều cấu trúc AI tiếng Việt ({vi_hits} pattern)")
    elif vi_hits >= 1:
        score += 10
        reasons.append(f"Có {vi_hits} cấu trúc câu kiểu AI")

    # ── 3. Đếm pattern AI tiếng Anh ─────────────────────────────────────────
    en_hits = sum(1 for p in CHATGPT_EN_PATTERNS if re.search(p, lower))
    if en_hits >= 3:
        score += 25
        reasons.append(f"Nhiều cụm từ AI tiếng Anh ({en_hits}): furthermore, moreover...")
    elif en_hits >= 1:
        score += 12
        reasons.append(f"Cụm từ AI tiếng Anh ({en_hits})")

    # ── 4. Mật độ từ hàn lâm "siêu an toàn" (AI rất thích) ─────────────────
    academic_count = sum(lower.count(w) for w in ACADEMIC_OVERUSE)
    words_total = len(lower.split())
    academic_density = academic_count / max(words_total, 1)
    if academic_density > 0.08:
        score += 20
        reasons.append(f"Mật độ từ học thuật rất cao ({academic_density:.1%}) — AI thường dùng")
    elif academic_density > 0.05:
        score += 10
        reasons.append(f"Nhiều từ học thuật chung chung ({academic_density:.1%})")

    # ── 5. Thiếu lỗi ngữ pháp / văn phong quá hoàn hảo ─────────────────────
    # Người thật hay có: "...", câu cảm thán, câu hỏi tu từ, dấu ngoặc
    human_signals = len(re.findall(r'[!?]', text)) + len(re.findall(r'\(.*?\)', text))
    if len(sents) > 5 and human_signals == 0:
        score += 12
        reasons.append("Không có câu hỏi/cảm thán/ngoặc — văn quá đều")

    # ── 6. Thiếu giọng cá nhân ───────────────────────────────────────────────
    personal = [r'\btôi\b', r'\bmình\b', r'\btheo (tôi|mình|chúng tôi)\b',
                r'\bchúng tôi (nhận|thấy|nghĩ)\b', r'\bkinh nghiệm (của|bản thân)\b']
    if not any(re.search(p, lower) for p in personal) and len(text) > 200:
        score += 10
        reasons.append("Không có giọng cá nhân/kinh nghiệm thực")

    # ── 7. Cấu trúc "định nghĩa + ví dụ + kết luận" lặp lại ─────────────────
    definition_patterns = [r'\blà (một|quá trình|phương pháp|cách)\b',
                           r'\bđược định nghĩa\b', r'\bcó thể hiểu là\b']
    def_count = sum(1 for p in definition_patterns if re.search(p, lower))
    if def_count >= 2:
        score += 8
        reasons.append("Nhiều câu định nghĩa — cấu trúc phổ biến của AI")

    # ── 8. Tỷ lệ unique words (perplexity proxy) ─────────────────────────────
    words = re.findall(r'\b\w{4,}\b', lower)
    if len(words) > 20:
        ur = len(set(words)) / len(words)
        if ur < 0.38:
            score += 18
            reasons.append(f"Từ vựng lặp nhiều (unique={ur:.0%}) — đặc trưng AI")
        elif ur < 0.48:
            score += 8
            reasons.append(f"Từ vựng hơi lặp (unique={ur:.0%})")

    return min(score, 100), reasons

def guess_ai_tool(text, score):
    if score < 35: return []
    lower = text.lower()
    tools = []
    # ChatGPT: thứ nhất/hai/ba, "tóm lại", "nhìn chung", câu dài đều
    if re.search(r'\bthứ (nhất|hai|ba)\b|\btóm lại\b|\bnhìn chung\b', lower):
        tools.append('ChatGPT / GPT-4')
    # Gemini: bullet, ngắn, "dưới đây là"
    if re.search(r'(\n[-•*]\s|dưới đây là|sau đây là)', lower):
        tools.append('Google Gemini')
    # Claude: dấu —, sắc thái, "điều này"
    if '—' in text or re.search(r'\bđiều này\b.*\b(cho thấy|gợi ý|có nghĩa)\b', lower):
        tools.append('Claude (Anthropic)')
    # Copilot
    if re.search(r'(microsoft|copilot|github|visual studio)', lower):
        tools.append('Microsoft Copilot')
    if not tools and score >= 55:
        tools = ['ChatGPT / GPT-4', 'Google Gemini', 'Claude']
    elif not tools and score >= 35:
        tools = ['ChatGPT / GPT-4', 'Google Gemini']
    return tools

# ─── Deep Analysis với Claude API ─────────────────────────────────────────────

def deep_analyze_with_claude(api_key, chunk):
    client = anthropic.Anthropic(api_key=api_key)
    try:
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=600,
            system="""Bạn là chuyên gia phát hiện văn bản do AI tạo ra với độ chính xác cao.
Phân tích kỹ đoạn văn và trả về JSON (CHỈ JSON, không có text khác):
{
  "ai_score": 0-100,
  "verdict": "AI_GENERATED|LIKELY_AI|UNCERTAIN|LIKELY_HUMAN|HUMAN",
  "ai_tools": ["tên công cụ cụ thể"],
  "reasons": ["lý do cụ thể 1", "lý do 2", "lý do 3"],
  "writing_style": "mô tả ngắn gọn về phong cách"
}

THANG ĐIỂM:
- 85-100: Chắc chắn AI (cấu trúc hoàn hảo, nhiều cụm nối, không lỗi, văn đều)  
- 65-84: Rất có thể AI
- 45-64: Có dấu hiệu AI, không chắc chắn
- 25-44: Có thể người viết, hoặc người dùng AI sửa nhẹ
- 0-24: Người viết (có lỗi nhỏ, giọng cá nhân, văn không đều)

DẤU HIỆU AI MẠNH:
- Burstiness thấp: các câu dài gần bằng nhau
- Cụm chuyển đoạn: "bên cạnh đó", "ngoài ra", "hơn nữa", "quan trọng là"
- Kết cấu hoàn hảo: intro→thân→kết không sai sót
- Từ "siêu an toàn": "hiệu quả", "toàn diện", "bền vững", "tích cực", "đổi mới"
- Thiếu giọng cá nhân, kinh nghiệm thực, cảm xúc
- Liệt kê có thứ tự: "thứ nhất... thứ hai... thứ ba"

NHẬN DIỆN CÔNG CỤ:
- ChatGPT: liệt kê thứ nhất/hai/ba, câu dài, kết bằng "tóm lại/nhìn chung"
- Gemini: bullet points, câu ngắn hơn, "dưới đây là danh sách"
- Claude: dùng dấu —, văn có sắc thái hơn, đôi khi hơi dài
- Copilot: ngữ cảnh kỹ thuật, code, tài liệu""",
            messages=[{"role": "user", "content": f"Phân tích đoạn văn sau và cho điểm AI:\n\n{chunk[:1400]}"}]
        )
        raw = msg.content[0].text
        clean = raw.replace('```json','').replace('```','').strip()
        return json.loads(clean)
    except Exception as e:
        return None

# ─── Plagiarism ───────────────────────────────────────────────────────────────

def search_google_serpapi(query, serpapi_key):
    try:
        resp = req_lib.get("https://serpapi.com/search", params={
            "q": f'"{query}"', "api_key": serpapi_key,
            "num": 5, "hl": "vi", "gl": "vn"
        }, timeout=8)
        results = resp.json().get("organic_results", [])[:3]
        return [{"title": r.get("title",""), "url": r.get("link",""), "snippet": r.get("snippet","")} for r in results]
    except: return []

def search_google_free(query):
    encoded = req_lib.utils.quote(f'"{query}"')
    return [{"title": "🔍 Tìm kiếm trên Google", "url": f"https://www.google.com/search?q={encoded}&hl=vi", "snippet": "Click để xác nhận thủ công"}]

def heuristic_plagiarism_score(text):
    score = 0
    lower = text.lower()
    official = [
        'theo quy định','căn cứ vào','nhằm mục đích','trong quá trình thực hiện',
        'được quy định tại','góp phần nâng cao','phát triển bền vững',
        'đề xuất giải pháp','qua thực tiễn cho thấy','từ những kết quả trên',
        'nhằm đáp ứng yêu cầu','trong giai đoạn hiện nay','chủ trương đường lối',
        'đảng và nhà nước','nêu trên','như sau','bao gồm các nội dung'
    ]
    score += min(sum(1 for p in official if p in lower) * 7, 45)
    if not re.search(r'\b(tôi|chúng tôi|chúng ta|bản thân)\b', lower): score += 8
    sents = [s for s in re.split(r'[.!?]', text) if s.strip()]
    long_s = [s for s in sents if len(s.split()) > 40]
    if len(long_s) / max(len(sents), 1) > 0.3: score += 12
    return min(score, 80)

# ─── Build highlighted document view ─────────────────────────────────────────

def find_best_pos(text, segment, start=0):
    """Tìm vị trí khớp tốt nhất của segment trong text từ vị trí start."""
    # 1. Khớp chính xác
    probe = segment[:120].strip()
    pos = text.find(probe, start)
    if pos != -1:
        return pos

    # 2. Thử chuỗi ngắn hơn
    for length in [80, 50, 30]:
        probe = segment[:length].strip()
        if len(probe) > 10:
            pos = text.find(probe, start)
            if pos != -1:
                return pos

    # 3. Khớp từng từ đầu tiên
    words = segment.split()[:8]
    if words:
        probe = ' '.join(words)
        pos = text.find(probe, start)
        if pos != -1:
            return pos

    return -1

def build_highlighted_document(main_text, results, no_overlap_chunks=None):
    """
    Map từng chunk phân tích ngược lại vào văn bản gốc.
    Dùng no_overlap_chunks để map chính xác hơn nếu có.
    Trả về list blocks: {text, type, chunk_idx, ai_score, plag_score, ...}
    """
    if not results:
        return [{"text": main_text, "type": "normal", "chunk_idx": -1,
                 "ai_score": 0, "plag_score": 0, "sources": []}]

    # Tạo lookup: text ngắn của no_overlap_chunk → index trong results
    # Mỗi no_overlap_chunk khớp với result gần nhất theo vị trí
    use_chunks = no_overlap_chunks if no_overlap_chunks else [r['segment'] for r in results]

    blocks = []
    cursor = 0
    text_len = len(main_text)

    for chunk_idx, seg_raw in enumerate(use_chunks):
        seg = seg_raw.strip()
        if not seg:
            continue

        # Tìm result tương ứng (theo chunk_idx hoặc closest match)
        if chunk_idx < len(results):
            r = results[chunk_idx]
        else:
            r = results[-1]

        pos = find_best_pos(main_text, seg, cursor)

        # Xác định loại highlight
        ai_s  = r.get('ai_score', 0)
        plag_s = r.get('plag_score', 0)
        if ai_s >= 60:
            btype = 'ai_high'
        elif ai_s >= 35:
            btype = 'ai_med'
        elif plag_s >= 50:
            btype = 'plag_high'
        elif plag_s >= 25:
            btype = 'plag_med'
        else:
            btype = 'normal'

        meta = {
            "chunk_idx": chunk_idx,
            "ai_score":  ai_s,
            "plag_score": plag_s,
            "verdict":   r.get('verdict', ''),
            "ai_tools":  r.get('ai_tools', []),
            "reasons":   r.get('reasons', []),
            "sources":   r.get('sources', []),
            "deep_analyzed": r.get('deep_analyzed', False),
        }

        if pos == -1:
            # Không tìm thấy vị trí chính xác — vẫn thêm block nhưng không có before
            blocks.append({"text": seg, "type": btype, **meta})
            continue

        # Phần trước đoạn → normal
        if pos > cursor:
            before_text = main_text[cursor:pos]
            if before_text.strip():
                blocks.append({"text": before_text, "type": "normal",
                               "chunk_idx": -1, "ai_score": 0, "plag_score": 0, "sources": []})

        # Tìm điểm kết thúc thực tế: dùng 50 ký tự cuối của segment để định vị
        seg_tail = seg[-50:].strip() if len(seg) > 50 else seg.strip()
        tail_pos = main_text.find(seg_tail, pos)
        if tail_pos != -1:
            end_pos = tail_pos + len(seg_tail)
        else:
            end_pos = min(pos + len(seg), text_len)

        # Không vượt quá văn bản
        end_pos = min(end_pos, text_len)
        actual_text = main_text[pos:end_pos]

        blocks.append({"text": actual_text, "type": btype, **meta})
        cursor = end_pos

    # Phần cuối còn lại
    if cursor < text_len:
        tail = main_text[cursor:].strip()
        if tail:
            blocks.append({"text": tail, "type": "normal",
                           "chunk_idx": -1, "ai_score": 0, "plag_score": 0, "sources": []})

    return blocks

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Hỗ trợ cả 2 chế độ: file upload và paste text
        main_text = ''
        template_text = ''
        excluded_ratio = 0.0

        input_mode = request.form.get('input_mode', 'file')  # 'file' hoặc 'text'

        if input_mode == 'text':
            main_text = request.form.get('main_text', '').strip()
            if len(main_text) < 30:
                return jsonify({"error": "Văn bản quá ngắn (tối thiểu 30 ký tự)"}), 400
        else:
            main_file = request.files.get('main_file')
            if not main_file:
                return jsonify({"error": "Thiếu file tài liệu chính"}), 400
            main_text = extract_text_from_file(main_file)
            if len(main_text) < 50:
                return jsonify({"error": "Tài liệu quá ngắn hoặc không đọc được"}), 400

        # File mẫu là tùy chọn
        template_file = request.files.get('template_file')
        if template_file and template_file.filename:
            template_text = extract_text_from_file(template_file)
            cleaned_text, excluded_ratio = remove_template_content(main_text, template_text)
            if len(cleaned_text) < 30:
                return jsonify({"error": "Sau khi loại trừ mẫu, không còn đủ nội dung"}), 400
        else:
            cleaned_text = main_text  # Không có mẫu → phân tích toàn bộ

        anthropic_key = request.form.get('anthropic_key', '').strip()
        serpapi_key = request.form.get('serpapi_key', '').strip()

        # Chunk toàn bộ văn bản
        chunks = chunk_text(cleaned_text, chunk_size=1500)
        total_chunks = len(chunks)
        if total_chunks == 0:
            return jsonify({"error": "Không thể chia nhỏ văn bản"}), 400

        # Heuristic scan TOÀN BỘ
        results = []
        for chunk in chunks:
            ai_score, reasons = heuristic_ai_score(chunk)
            plag_score = heuristic_plagiarism_score(chunk)
            tools = guess_ai_tool(chunk, ai_score)
            verdict = ("AI_GENERATED" if ai_score >= 70 else
                      "LIKELY_AI" if ai_score >= 50 else
                      "UNCERTAIN" if ai_score >= 30 else "LIKELY_HUMAN")
            results.append({
                "segment": chunk,
                "ai_score": ai_score,
                "plag_score": plag_score,
                "verdict": verdict,
                "ai_tools": tools,
                "reasons": reasons,
                "writing_style": "Heuristic",
                "deep_analyzed": False,
                "sources": []
            })

        # Deep analysis với Claude — chỉ cho đoạn nghi vấn cao
        deep_count = 0
        if anthropic_key:
            suspicious = sorted(
                [(i, r) for i, r in enumerate(results) if r['ai_score'] >= 40],
                key=lambda x: x[1]['ai_score'], reverse=True
            )[:30]
            for idx, r in suspicious:
                deep = deep_analyze_with_claude(anthropic_key, r['segment'])
                if deep:
                    results[idx].update({
                        'ai_score': deep.get('ai_score', r['ai_score']),
                        'verdict': deep.get('verdict', r['verdict']),
                        'ai_tools': deep.get('ai_tools', r['ai_tools']),
                        'reasons': deep.get('reasons', r['reasons']),
                        'writing_style': deep.get('writing_style', ''),
                        'deep_analyzed': True
                    })
                    deep_count += 1

        # Plagiarism sources
        for r in results:
            query_text = r['segment'][:120].strip()
            if serpapi_key and r['plag_score'] > 30:
                sources = search_google_serpapi(query_text, serpapi_key)
                if sources: r['plag_score'] = min(r['plag_score'] + 15, 95)
            else:
                sources = search_google_free(query_text)
            r['sources'] = sources

        avg_ai = round(sum(r['ai_score'] for r in results) / total_chunks)
        avg_plag = round(sum(r['plag_score'] for r in results) / total_chunks)

        # Tạo document view có highlight (dùng chunk không overlap để map chính xác)
        no_overlap_chunks = chunk_text_no_overlap(cleaned_text, chunk_size=1500)
        doc_blocks = build_highlighted_document(cleaned_text, results, no_overlap_chunks)

        return jsonify({
            "success": True,
            "stats": {
                "total_chars": len(main_text),
                "cleaned_chars": len(cleaned_text),
                "excluded_ratio": round(excluded_ratio * 100, 1),
                "chunks_analyzed": total_chunks,
                "deep_analyzed": deep_count,
                "has_template": bool(template_file and template_file.filename),
                "input_mode": input_mode,
                "method_ai": f"Heuristic + Claude deep ({deep_count} đoạn nghi vấn)" if anthropic_key else "Heuristic 8 chiều",
                "method_plag": "SerpAPI ✓ (tự động tìm nguồn)" if serpapi_key else "Heuristic + link Google"
            },
            "scores": {"ai_pct": avg_ai, "plag_pct": avg_plag},
            "ai_results": results,
            "plag_results": results,
            "doc_blocks": doc_blocks
        })

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    from werkzeug.serving import WSGIRequestHandler
    WSGIRequestHandler.protocol_version = "HTTP/1.1"
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
