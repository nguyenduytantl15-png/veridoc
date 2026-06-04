import os, re, json, math, difflib, statistics
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge
import anthropic
import mammoth
import requests as req_lib

app = Flask(__name__)

# ─── Advanced NLP helpers ──────────────────────────────────────────────────────

def calc_text_entropy(text):
    """Tính entropy Shannon của văn bản — AI có entropy thấp hơn người thật."""
    import math
    if not text: return 0.0
    words = re.findall(r'\b\w+\b', text.lower())
    if len(words) < 5: return 0.0
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    total = len(words)
    entropy = -sum((c/total) * math.log2(c/total) for c in freq.values())
    return round(entropy, 3)

def calc_ttr(text):
    """Type-Token Ratio — AI thường lặp từ nhiều hơn (TTR thấp hơn)."""
    words = re.findall(r'\b\w{3,}\b', text.lower())
    if not words: return 1.0
    return len(set(words)) / len(words)

def calc_avg_word_length(text):
    """AI thường dùng từ dài hơn, hàn lâm hơn."""
    words = re.findall(r'\b\w+\b', text)
    if not words: return 0.0
    return sum(len(w) for w in words) / len(words)

def calc_punctuation_diversity(text):
    """Người thật dùng dấu câu đa dạng hơn (!?...()), AI dùng ít dấu đặc biệt."""
    special = len(re.findall(r'[!?\.()\'\"\-—…]', text))
    total = max(len(text), 1)
    return special / total

def calc_sentence_start_diversity(text):
    """AI thường bắt đầu câu bằng cùng một số pattern, người thật đa dạng hơn."""
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 5]
    if len(sents) < 3: return 1.0
    # Lấy 2 từ đầu của mỗi câu
    starts = []
    for s in sents:
        words = s.split()[:2]
        starts.append(' '.join(w.lower() for w in words))
    return len(set(starts)) / len(starts)  # Cao = đa dạng = người thật

def calc_comma_ratio(text):
    """AI thường dùng dấu phẩy nhiều trong câu phức."""
    sentences = [s for s in re.split(r'[.!?]', text) if s.strip()]
    if not sentences: return 0.0
    commas_per_sent = sum(s.count(',') for s in sentences) / len(sentences)
    return commas_per_sent


CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024

@app.errorhandler(RequestEntityTooLarge)

def calc_ai_verb_density(text):
    """AI ưa dùng động từ hành chính: đóng vai trò, mang lại, góp phần, triển khai..."""
    ai_verbs = [
        'đóng vai trò', 'mang lại', 'góp phần', 'tạo điều kiện', 'phát huy',
        'nâng cao', 'triển khai', 'thực hiện hiệu quả', 'hình thành',
        'xây dựng được', 'đảm bảo', 'thúc đẩy', 'tăng cường',
        'cải thiện', 'phát triển toàn diện', 'nâng tầm',
    ]
    lower = text.lower()
    count = sum(lower.count(v) for v in ai_verbs)
    sents = max(len(re.split(r'[.!?]+', text)), 1)
    return count / sents

def calc_concrete_detail_score(text):
    """Người thật hay có: số liệu, tên riêng, ngày tháng, lớp cụ thể — AI hay chung chung."""
    # Số liệu: %, con số, năm học
    numbers = len(re.findall(r'\d+(?:[.,]\d+)?%?', text))
    # Tên riêng/địa danh (chữ hoa giữa câu)
    proper_nouns = len(re.findall(r'(?<=[^A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐƠƯ])[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐƠƯ][a-zàáâãèéêìíòóôõùúýăđơư]+', text))
    sents = max(len(re.split(r'[.!?]+', text)), 1)
    # Normalize
    score = (numbers * 2 + proper_nouns * 0.5) / sents
    return min(score, 5.0)

def calc_ai_sentence_endings(text):
    """AI thường kết câu bằng tính từ hàn lâm tích cực."""
    ai_endings = [
        'tích cực', 'hiệu quả', 'toàn diện', 'bền vững', 'sáng tạo',
        'đáng kể', 'rộng rãi', 'đúng đắn', 'quan trọng', 'thiết yếu',
        'cần thiết', 'phù hợp', 'đồng bộ', 'thiết thực', 'chủ động',
        'tốt hơn', 'mạnh mẽ', 'rõ rệt', 'đáng khen', 'có giá trị',
    ]
    sents = [s.strip().lower() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 5]
    if not sents: return 0.0
    hits = sum(1 for s in sents if any(s.endswith(e) for e in ai_endings))
    return hits / len(sents)


def too_large(e):
    return jsonify({"error": "File quá lớn! Vui lòng dùng file dưới 200MB."}), 413

os.makedirs('uploads', exist_ok=True)

# ─── Text Extraction ──────────────────────────────────────────────────────────

def extract_text_from_file(file_storage, return_html=False):
    """
    Đọc file và trả về text thuần (cho analysis) hoặc HTML (cho hiển thị).
    return_html=True → giữ định dạng heading, bold, paragraph.
    """
    filename = file_storage.filename.lower()
    content = file_storage.read()

    if filename.endswith('.txt'):
        raw = content.decode('utf-8', errors='ignore')
        if return_html:
            # Chuyển xuống dòng thành <p>
            paras = [p.strip() for p in raw.split('\n\n') if p.strip()]
            html = ''.join(f'<p>{p.replace(chr(10), "<br>")}</p>' for p in paras)
            return raw, html
        return raw

    elif filename.endswith('.pdf'):
        raw = extract_pdf(content)
        if return_html:
            paras = [p.strip() for p in raw.split('\n\n') if p.strip()]
            html = ''.join(f'<p>{p.replace(chr(10), "<br>")}</p>' for p in paras)
            return raw, html
        return raw

    elif filename.endswith('.docx'):
        import io
        if return_html:
            # Giữ định dạng: heading, bold, italic, paragraph
            result_html = mammoth.convert_to_html(io.BytesIO(content))
            result_text = mammoth.extract_raw_text(io.BytesIO(content))
            return result_text.value, result_html.value
        return mammoth.extract_raw_text(io.BytesIO(content)).value

    elif filename.endswith('.doc'):
        import io, tempfile
        def try_extract(src):
            if return_html:
                rh = mammoth.convert_to_html(src)
                rt = mammoth.extract_raw_text(src)
                return rt.value, rh.value
            return mammoth.extract_raw_text(src).value

        try:
            r = try_extract(io.BytesIO(content))
            text = r[0] if return_html else r
            if text.strip():
                return r
        except: pass
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.doc', delete=False) as tmp:
                tmp.write(content); tmp_path = tmp.name
            with open(tmp_path, 'rb') as f:
                return try_extract(f)
        except:
            raw = content.decode('utf-8', errors='ignore')
            if return_html: return raw, f'<p>{raw}</p>'
            return raw
        finally:
            if tmp_path:
                try: os.unlink(tmp_path)
                except: pass

    raw = content.decode('utf-8', errors='ignore')
    if return_html:
        return raw, f'<p>{raw}</p>'
    return raw

def extract_pdf(content):
    """Đọc text từ PDF, hỗ trợ cả PDF thường và scan (OCR cơ bản)."""
    import io, tempfile
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            pages = []
            for page in pdf.pages:
                text = page.extract_text()
                if text and text.strip():
                    pages.append(text.strip())
            result = '\n\n'.join(pages)
            if result.strip():
                return result
    except Exception as e:
        pass
    # Fallback: dùng fitz (PyMuPDF) nếu pdfplumber thất bại
    try:
        import fitz
        doc = fitz.open(stream=content, filetype='pdf')
        pages = [doc[i].get_text() for i in range(len(doc))]
        return '\n\n'.join(p for p in pages if p.strip())
    except:
        pass
    return '[Không đọc được nội dung PDF. File có thể bị mã hóa hoặc chỉ chứa hình ảnh.]' 

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

# ─── AI Detection Engine v2 (Nâng cấp độ chính xác) ─────────────────────────

# Pattern AI tiếng Việt — mở rộng và chính xác hơn
CHATGPT_VI_PATTERNS = [
    # Mở đầu đoạn kiểu AI — rất đặc trưng
    r'\btất nhiên[,\s]', r'\bchắc chắn rằng\b', r'\bkhông thể phủ nhận\b',
    r'\btrong bối cảnh[\s]', r'\btrong thời đại[\s]', r'\bngày nay[,\s]',
    r'\bnhư chúng ta (đã )?biết\b', r'\bcó thể (thấy|nói|khẳng định) rằng\b',
    r'\bđiều này cho thấy\b', r'\bđiều đó có nghĩa\b',
    r'\bvấn đề (này|đó|trên) (là|cần|đòi hỏi)\b',
    r'\bthực tiễn cho thấy\b', r'\bqua thực tiễn\b',
    r'\bđây là (một|điều|vấn đề|xu hướng)\b',
    r'\bcó thể (thấy|nhận thấy|nhận ra)\b',
    # Kết luận AI
    r'\btóm lại[,\s]', r'\bnhìn chung[,\s]', r'\bkết luận lại\b',
    r'\bqua (những|các) (phân tích|nghiên cứu|ví dụ) trên\b',
    r'\btừ (những|các) (điều|vấn đề|phân tích) (trên|nêu trên)\b',
    r'\bnhìn lại (quá trình|việc|những)\b',
    r'\bđể (kết luận|tóm tắt|tổng kết)\b',
    # Chuyển đoạn AI — rất phổ biến
    r'\bbên cạnh đó[,\s]', r'\bngoài ra[,\s]', r'\bhơn nữa[,\s]',
    r'\bđồng thời[,\s]', r'\bthêm vào đó[,\s]', r'\btrong khi đó[,\s]',
    r'\bngược lại[,\s]', r'\bdo đó[,\s]', r'\bvì vậy[,\s]',
    r'\btheo đó[,\s]', r'\btuy nhiên[,\s].*\bnhưng\b',
    # Nhấn mạnh AI
    r'\bquan trọng (là|hơn|nhất)[,\s]', r'\bcần lưu ý (rằng|là|đây)[\s]',
    r'\bđáng chú ý (là|rằng)[\s]', r'\bđặc biệt (là|quan trọng)[\s]',
    r'\bđây là (một|điều|vấn đề|yếu tố) (quan trọng|thiết yếu|cần thiết)\b',
    r'\bkhông thể (thiếu|bỏ qua|phủ nhận)\b',
    # Liệt kê AI
    r'\bthứ (nhất|hai|ba|tư|năm)\b',
    r'\b(đầu tiên|tiếp theo|cuối cùng)[,\s]',
    r'\bmột (là|mặt)[,\s].*\bhai (là|mặt)\b',
    r'\bcác (yếu tố|nguyên nhân|giải pháp|bước) (sau|bao gồm|chính)\b',
    # Câu định nghĩa AI
    r'\b(được hiểu là|được định nghĩa là|có thể hiểu là)\b',
    r'\blà (quá trình|phương pháp|cách|hệ thống) [a-zA-ZÀ-ỹ]',
    # Cụm "siêu hàn lâm" AI
    r'\bphát triển bền vững\b', r'\bnâng cao chất lượng\b',
    r'\bđổi mới (phương pháp|cách tiếp cận|tư duy)\b',
    r'\btriển khai (hiệu quả|đồng bộ|toàn diện)\b',
    r'\bgiải pháp (toàn diện|hiệu quả|phù hợp)\b',
    r'\bnhằm (mục đích|đáp ứng|nâng cao|phát triển)\b',
    r'\btrong (bối cảnh|giai đoạn|thời kỳ) (hiện nay|hiện tại|ngày nay)\b',
]

CHATGPT_EN_PATTERNS = [
    r'\bfirstly\b.*\bsecondly\b', r'\bin conclusion\b', r'\bin summary\b',
    r'\bfurthermore\b', r'\bmoreover\b', r'\badditionally\b', r'\bnotably\b',
    r'\bit is worth noting\b', r'\bit is important to\b', r'\bsignificantly\b',
    r'\bthis highlights\b', r'\bthis demonstrates\b', r'\bthis suggests\b',
    r'\bin other words\b', r'\bto summarize\b', r'\boverall\b',
    r'\bas a result\b', r'\bconsequently\b', r'\bit should be noted\b',
    r'\bwith regard to\b', r'\bin terms of\b', r'\bwhen it comes to\b',
]

# Từ "siêu an toàn" AI hay dùng — mở rộng
ACADEMIC_OVERUSE = [
    'toàn diện','hiệu quả','tích cực','bền vững','phát triển','nâng cao',
    'đổi mới','triển khai','thực hiện','đảm bảo','chất lượng','năng lực',
    'hệ thống','quá trình','phương pháp','giải pháp','chiến lược','mục tiêu',
    'kết quả','đánh giá','cần thiết','quan trọng','phù hợp','đồng bộ',
    'thiết thực','cụ thể','rõ ràng','minh bạch','chủ động','sáng tạo',
    'comprehensive','effective','significant','important','various','essential',
    'crucial','innovative','sustainable','optimize','furthermore','moreover',
]

# Dấu hiệu người thật (giảm điểm AI nếu có)
HUMAN_SIGNALS = [
    r'\btôi\b', r'\bmình\b', r'\btheo tôi\b', r'\bchúng tôi (nhận|thấy|nghĩ)\b',
    r'\btừ kinh nghiệm\b', r'\bthật ra\b', r'\bthực ra\b', r'\bthú thật\b',
    r'\bnhớ lại\b', r'\bhôm nay\b', r'\bngày hôm đó\b',
    r'[!?]{1,}', r'\.\.\.',  # dấu câu cảm xúc
    r'\(.*?\)',  # ngoặc đơn
    r'\bví dụ (như|cụ thể)\b',  # ví dụ cụ thể
    r'"[^"]{3,50}"',  # trích dẫn
]

def heuristic_ai_score(text):
    score = 0
    reasons = []
    lower = text.lower()
    sents = [s.strip() for s in re.split(r'[.!?।]+', text) if len(s.strip()) > 8]
    if not sents:
        return 0, []

    # ── 1. Burstiness (CV câu) ───────────────────────────────────────────────
    lengths = [len(s.split()) for s in sents]
    if len(lengths) >= 3:
        try:
            import statistics as _stat
            stdev = _stat.stdev(lengths)
            mean  = _stat.mean(lengths)
            cv = stdev / mean if mean > 0 else 1
            if cv < 0.20:
                score += 35
                reasons.append(f"Câu rất đều nhau (CV={cv:.2f}) — đặc trưng rõ nhất của AI")
            elif cv < 0.30:
                score += 22
                reasons.append(f"Câu khá đồng đều (CV={cv:.2f})")
            elif cv < 0.40:
                score += 10
                reasons.append(f"Câu hơi đều (CV={cv:.2f})")
        except: pass

    # ── 2. Đếm pattern AI tiếng Việt ────────────────────────────────────────
    vi_hits = []
    for p in CHATGPT_VI_PATTERNS:
        try:
            if re.search(p, lower): vi_hits.append(p)
        except: pass
    if len(vi_hits) >= 8:
        score += 40
        reasons.append(f"Rất nhiều cấu trúc AI tiếng Việt ({len(vi_hits)} pattern)")
    elif len(vi_hits) >= 5:
        score += 28
        reasons.append(f"Nhiều cấu trúc AI ({len(vi_hits)} pattern): bên cạnh đó, nhìn chung...")
    elif len(vi_hits) >= 3:
        score += 18
        reasons.append(f"Có {len(vi_hits)} cấu trúc câu kiểu AI")
    elif len(vi_hits) >= 1:
        score += 8

    # ── 3. Pattern AI tiếng Anh ─────────────────────────────────────────────
    en_hits = sum(1 for p in CHATGPT_EN_PATTERNS if re.search(p, lower))
    if en_hits >= 4:
        score += 28
        reasons.append(f"Nhiều cụm AI tiếng Anh ({en_hits}): furthermore, moreover...")
    elif en_hits >= 2:
        score += 15
        reasons.append(f"Cụm từ AI tiếng Anh ({en_hits})")
    elif en_hits >= 1:
        score += 7

    # ── 4. Mật độ từ hàn lâm "siêu an toàn" ────────────────────────────────
    words_total = max(len(lower.split()), 1)
    academic_count = sum(lower.count(w) for w in ACADEMIC_OVERUSE)
    academic_density = academic_count / words_total
    if academic_density > 0.10:
        score += 25
        reasons.append(f"Mật độ từ học thuật rất cao ({academic_density:.1%}) — AI thường dùng")
    elif academic_density > 0.06:
        score += 14
        reasons.append(f"Nhiều từ học thuật ({academic_density:.1%})")
    elif academic_density > 0.04:
        score += 6

    # ── 5. Thiếu dấu hiệu người thật ────────────────────────────────────────
    human_count = 0
    for p in HUMAN_SIGNALS:
        try:
            if re.search(p, text): human_count += 1
        except: pass
    if human_count == 0 and len(text) > 200:
        score += 15
        reasons.append("Không có dấu hiệu người thật (cảm xúc, tôi, ví dụ, dấu câu đặc biệt)")
    elif human_count >= 3:
        score = max(0, score - 12)
        reasons.append(f"Có {human_count} dấu hiệu giọng người thật → giảm điểm")

    # ── 6. Tỷ lệ unique words (perplexity proxy) ────────────────────────────
    words = re.findall(r'\b\w{4,}\b', lower)
    if len(words) > 30:
        ur = len(set(words)) / len(words)
        if ur < 0.35:
            score += 20
            reasons.append(f"Từ vựng lặp nhiều (unique={ur:.0%}) — AI dùng từ ít đa dạng")
        elif ur < 0.45:
            score += 10
            reasons.append(f"Từ vựng hơi lặp (unique={ur:.0%})")
        elif ur > 0.70:
            score = max(0, score - 8)  # Người thật dùng từ đa dạng hơn

    # ── 7. Câu định nghĩa lặp lại ───────────────────────────────────────────
    def_patterns = [
        r'\blà (một|quá trình|phương pháp|cách|hệ thống|yếu tố)\b',
        r'\bđược (định nghĩa|hiểu|xem) là\b',
        r'\bcó thể (hiểu|xem) là\b'
    ]
    def_count = sum(1 for p in def_patterns if re.search(p, lower))
    if def_count >= 3:
        score += 10
        reasons.append("Nhiều câu định nghĩa liên tiếp — phong cách AI")

    # ── 8. Độ dài câu trung bình quá đều và dài ─────────────────────────────
    if lengths:
        avg_len = sum(lengths) / len(lengths)
        long_ratio = sum(1 for l in lengths if l > 25) / len(lengths)
        if avg_len > 20 and long_ratio > 0.5:
            score += 10
            reasons.append(f"Câu dài và đều ({avg_len:.0f} từ/câu trung bình)")


    # ── 9. Entropy Shannon (AI viết đều đặn → entropy thấp) ──────────────────
    entropy = calc_text_entropy(text)
    if entropy > 0:
        if entropy < 3.5:
            score += 20
            reasons.append(f"Entropy văn bản thấp ({entropy}) — AI viết rất đều đặn")
        elif entropy < 4.2:
            score += 10
            reasons.append(f"Entropy hơi thấp ({entropy})")
        elif entropy > 5.0:
            score = max(0, score - 8)

    # ── 10. Type-Token Ratio ───────────────────────────────────────────────────
    ttr = calc_ttr(text)
    if ttr < 0.40:
        score += 15
        reasons.append(f"TTR thấp ({ttr:.2f}) — từ vựng ít đa dạng")
    elif ttr < 0.52:
        score += 7
    elif ttr > 0.72:
        score = max(0, score - 10)

    # ── 11. Độ đa dạng mở đầu câu ─────────────────────────────────────────────
    start_div = calc_sentence_start_diversity(text)
    if start_div < 0.50:
        score += 12
        reasons.append(f"Câu bắt đầu giống nhau ({start_div:.0%} đa dạng) — AI")
    elif start_div > 0.85:
        score = max(0, score - 8)

    # ── 12. Dấu câu đặc biệt ──────────────────────────────────────────────────
    punct_div = calc_punctuation_diversity(text)
    if punct_div < 0.02 and len(text) > 200:
        score += 10
        reasons.append("Gần như không có dấu câu đặc biệt (!?...) — AI ít dùng")
    elif punct_div > 0.06:
        score = max(0, score - 6)

    # ── 13. Chiều dài từ trung bình ───────────────────────────────────────────
    avg_wl = calc_avg_word_length(text)
    if avg_wl > 5.5:
        score += 8
        reasons.append(f"Từ trung bình dài ({avg_wl:.1f} ký tự) — ngôn ngữ hàn lâm")

    # ── 15. Mật độ AI-verbs (đóng vai trò, mang lại, góp phần...) ─────────────
    ai_verb_den = calc_ai_verb_density(text)
    if ai_verb_den >= 0.6:
        score += 22
        reasons.append(f"Mật độ động từ hành chính AI cao ({ai_verb_den:.1f}/câu): đóng vai trò, mang lại...")
    elif ai_verb_den >= 0.3:
        score += 12
        reasons.append(f"Có nhiều động từ hành chính kiểu AI ({ai_verb_den:.1f}/câu)")
    elif ai_verb_den >= 0.1:
        score += 5

    # ── 16. Thiếu chi tiết cụ thể (số liệu, tên, ngày tháng) ────────────────
    concrete = calc_concrete_detail_score(text)
    if concrete < 0.3 and len(text) > 150:
        score += 18
        reasons.append("Không có số liệu/tên cụ thể — AI thường viết chung chung")
    elif concrete < 0.8:
        score += 8
    elif concrete >= 2.0:
        score = max(0, score - 15)
        reasons.append(f"Có nhiều chi tiết cụ thể ({concrete:.1f}/câu) — dấu hiệu người viết")

    # ── 17. Kết câu bằng tính từ hàn lâm AI ─────────────────────────────────
    ae_ratio = calc_ai_sentence_endings(text)
    if ae_ratio >= 0.5:
        score += 18
        reasons.append(f"Nhiều câu kết bằng tính từ hàn lâm ({ae_ratio:.0%}): hiệu quả, toàn diện, bền vững...")
    elif ae_ratio >= 0.25:
        score += 8
        reasons.append(f"Một số câu kết bằng tính từ hàn lâm ({ae_ratio:.0%})")

    # ── 14. Ensemble — tổng hợp tất cả chỉ số ────────────────────────────────
    ai_signals = sum([
        1 if len(vi_hits) >= 3 else 0,
        1 if en_hits >= 2 else 0,
        1 if academic_density > 0.06 else 0,
        1 if human_count == 0 else 0,
        1 if entropy < 4.0 else 0,
        1 if ttr < 0.50 else 0,
        1 if start_div < 0.60 else 0,
        1 if ai_verb_den >= 0.3 else 0,
        1 if concrete < 0.5 else 0,
        1 if ae_ratio >= 0.25 else 0,
    ])
    if ai_signals >= 7:
        score += 18
        reasons.append(f"Ensemble: {ai_signals}/10 chỉ số AI — bằng chứng rất mạnh")
    elif ai_signals >= 5:
        score += 12
        reasons.append(f"Ensemble: {ai_signals}/10 chỉ số AI — nhiều bằng chứng")
    elif ai_signals >= 3:
        score += 5

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
    """Tìm kiếm Google qua SerpAPI — trả về URL, snippet, và highlight words."""
    try:
        resp = req_lib.get("https://serpapi.com/search", params={
            "q": f'"{query}"', "api_key": serpapi_key,
            "num": 5, "hl": "vi", "gl": "vn"
        }, timeout=10)
        data = resp.json()
        results = []
        q_words = [w for w in query.lower().split() if len(w) > 3][:8]
        for r in data.get("organic_results", [])[:4]:
            url   = r.get("link", "")
            title = r.get("title", "")
            snip  = r.get("snippet", "")
            # Tính độ khớp
            combined = (title + " " + snip).lower()
            match_words = sum(1 for w in q_words if w in combined)
            confidence = "high" if match_words >= 4 else ("medium" if match_words >= 2 else "low")
            # Tìm vị trí đoạn trùng trong snippet để highlight
            highlight_words = [w for w in q_words if w in combined]
            if url:
                results.append({
                    "title":           title,
                    "url":             url,
                    "snippet":         snip,
                    "confidence":      confidence,
                    "highlight_words": highlight_words,
                    "source":          "google"
                })
        return results
    except Exception as e:
        return []

def search_google_free_fetch(query):
    """Không có SerpAPI: trả link Google + danh sách từ khóa highlight."""
    encoded = req_lib.utils.quote(f'"{query}"')
    highlight_words = [w for w in query.lower().split() if len(w) > 3][:6]
    return [{
        "title":           "🔍 Tìm trên Google (click để kiểm tra thủ công)",
        "url":             f"https://www.google.com/search?q={encoded}&hl=vi",
        "snippet":         "Nhấn để mở Google và kiểm tra nguồn gốc đoạn văn này.",
        "confidence":      "manual",
        "highlight_words": highlight_words,
        "source":          "google_manual"
    }]

# search_google_free replaced by search_google_free_fetch above

def heuristic_plagiarism_score(text):
    """
    Đánh giá khả năng đạo văn — v3.
    Phát hiện: copy báo, copy Wikipedia, copy tài liệu học thuật/hành chính.
    """
    import statistics as _st
    score = 0
    lower = text.lower()
    sents = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
    n_sents = max(len(sents), 1)
    n_words = max(len(text.split()), 1)
    n_chars = max(len(text), 1)

    # ── 1. Cụm từ báo chí đặc trưng ─────────────────────────────────────────
    NEWS_PHRASES = [
        'theo báo cáo','theo phóng viên','được biết','liên quan đến',
        'cho biết thêm','theo nguồn tin','trao đổi với','chia sẻ với',
        'phát biểu tại','ghi nhận tại','theo ghi nhận','tính đến',
        'trong khi đó','trước đó','sau đó','ngay sau khi',
        'các chuyên gia cho rằng','theo các chuyên gia','giới chuyên môn',
        'theo nghiên cứu','kết quả nghiên cứu','nghiên cứu cho thấy',
        'theo số liệu','theo thống kê','khảo sát cho thấy',
        'đã được công bố','được ghi nhận','đã chứng minh',
    ]
    news_hits = sum(1 for p in NEWS_PHRASES if p in lower)
    if news_hits >= 5: score += 35
    elif news_hits >= 3: score += 25
    elif news_hits >= 2: score += 15
    elif news_hits >= 1: score += 8

    # ── 2. Cụm từ hành chính/văn phòng ──────────────────────────────────────
    OFFICIAL_PHRASES = [
        'theo quy định','căn cứ vào','nhằm mục đích','trong quá trình',
        'được quy định tại','nhằm đáp ứng','trong giai đoạn',
        'chủ trương','đảng và nhà nước','nêu trên','bao gồm các',
        'được tiến hành','theo đó','từ đó cho thấy',
        'đề xuất giải pháp','qua thực tiễn','từ những kết quả',
        'nhằm góp phần','đóng góp tích cực','triển khai đồng bộ',
        'thực hiện có hiệu quả','hoàn thành xuất sắc',
        'đạt và vượt chỉ tiêu','hướng tới mục tiêu',
    ]
    official_hits = sum(1 for p in OFFICIAL_PHRASES if p in lower)
    if official_hits >= 5: score += 30
    elif official_hits >= 3: score += 20
    elif official_hits >= 1: score += 8

    # ── 3. Cụm Wikipedia / bách khoa ─────────────────────────────────────────
    WIKI_PHRASES = [
        'là một quốc gia','là một thành phố','là một tổ chức',
        'được thành lập vào','có trụ sở tại','thuộc về',
        'nằm ở phía','giáp với','diện tích khoảng',
        'dân số khoảng','tính đến năm','vào năm',
        'được chia thành','gồm có','hay còn gọi là',
        'còn được biết đến','không chính thức','chính thức là',
        'là thủ đô','thành phố lớn nhất','là quốc gia',
    ]
    wiki_hits = sum(1 for p in WIKI_PHRASES if p in lower)
    if wiki_hits >= 4: score += 35
    elif wiki_hits >= 2: score += 22
    elif wiki_hits >= 1: score += 10

    # ── 4. Số liệu chính xác nhiều ───────────────────────────────────────────
    # Người thật ít có nhiều số liệu, văn copy/báo có nhiều
    pct_numbers = len(re.findall(r'\d+[,.]?\d*\s*%', text))
    year_numbers = len(re.findall(r'(19|20)\d{2}', text))
    big_numbers = len(re.findall(r'\d{1,3}(?:[.,]\d{3})+', text))
    unit_numbers = len(re.findall(r'\d+\s*(tỷ|triệu|nghìn|tỉ|USD|VNĐ|đồng|km|ha|m²)', text.lower()))
    total_numbers = pct_numbers * 3 + year_numbers + big_numbers * 2 + unit_numbers * 2
    if total_numbers >= 8: score += 25
    elif total_numbers >= 4: score += 15
    elif total_numbers >= 2: score += 8

    # ── 5. Thiếu giọng cá nhân ───────────────────────────────────────────────
    has_personal = bool(re.search(r'(tôi|chúng tôi|chúng ta|bản thân|mình|theo tôi)', lower))
    if not has_personal:
        score += 15
    else:
        score = max(0, score - 10)  # Có giọng cá nhân → giảm điểm

    # ── 6. Trích dẫn trực tiếp ───────────────────────────────────────────────
    quotes = len(re.findall(r'[""''«»„"‟❝❞]', text))
    if quotes >= 6: score += 15
    elif quotes >= 2: score += 7

    # ── 7. Tên tổ chức/cơ quan chính thức ────────────────────────────────────
    ORG_PATTERNS = [
        r'bộ \w+ và \w+', r'ủy ban \w+', r'hội đồng \w+',
        r'sở \w+ và \w+', r'phòng \w+', r'ban \w+',
        r'tổng cục \w+', r'cục \w+', r'vụ \w+',
        r'viện \w+', r'trường \w+', r'trung tâm \w+',
    ]
    org_hits = sum(1 for p in ORG_PATTERNS if re.search(p, lower))
    if org_hits >= 3: score += 12
    elif org_hits >= 1: score += 5

    # ── 8. Câu bị động chính thức ────────────────────────────────────────────
    passive_official = len(re.findall(
        r'được\s+\w+\s*(ký|ban hành|phê duyệt|công bố|thông qua|'
        r'triển khai|thực hiện|đánh giá|ghi nhận|xác nhận|công nhận)', lower))
    if passive_official >= 3: score += 12
    elif passive_official >= 1: score += 5

    # ── 9. Văn phong trung lập hoàn toàn (không cảm xúc) ─────────────────────
    EMOTION_WORDS = ['thiệt','thật ra','mà','cơ mà','nên','kỳ','ơi','ừ','nhỉ',
                     'vui','buồn','chán','sợ','hy vọng','lo','đau','mừng']
    has_emotion = any(w in lower for w in EMOTION_WORDS)
    if not has_emotion and n_words > 50:
        score += 8

    # ── 10. Độ đồng đều câu ──────────────────────────────────────────────────
    if len(sents) >= 3:
        lens = [len(s.split()) for s in sents]
        try:
            cv = _st.stdev(lens) / max(_st.mean(lens), 1)
            if cv < 0.25: score += 8
        except: pass

    return min(score, 95)

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


def build_html_doc_blocks(html_content, plain_text, results):
    """
    Tạo HTML đã highlight từ HTML gốc Word.
    Chèn highlight span vào đúng vị trí các đoạn nghi vấn.
    """
    import html as html_lib
    if not html_content:
        return None

    # Map chunk_idx → màu highlight
    chunk_colors = {}
    for r in results:
        cidx = r.get('chunk_idx', -1)
        if cidx < 0: continue
        ai_s = r.get('ai_score', 0)
        plag_s = r.get('plag_score', 0)
        if ai_s >= 60:     chunk_colors[cidx] = ('hl-ai-high', r)
        elif ai_s >= 35:   chunk_colors[cidx] = ('hl-ai-med', r)
        elif plag_s >= 50: chunk_colors[cidx] = ('hl-plag-high', r)
        elif plag_s >= 25: chunk_colors[cidx] = ('hl-plag-med', r)

    if not chunk_colors:
        return html_content  # Không có gì cần highlight

    # Highlight từng đoạn trong HTML bằng cách tìm text và bọc span
    result_html = html_content
    for cidx, (cls, r) in chunk_colors.items():
        seg = r.get('segment', '').strip()
        if not seg or len(seg) < 20: continue
        # Tìm 60 ký tự đầu của segment trong HTML (đã escape)
        probe = html_lib.escape(seg[:60])
        pos = result_html.find(probe)
        if pos == -1:
            # Thử không escape
            pos = result_html.find(seg[:40])
        if pos == -1: continue

        num = cidx + 1
        badge = f'<span class="seg-badge {cls.replace("hl-","sb-")}" onclick="onHlClickByChunk({cidx})">{num}</span>'
        # Bọc đoạn bắt đầu từ pos
        end = min(pos + len(seg) + 50, len(result_html))
        # Tìm điểm kết thúc thẻ đoạn gần nhất
        tag_end = result_html.find('</p>', pos)
        if tag_end == -1: tag_end = end
        # Bọc span highlight quanh nội dung
        result_html = (result_html[:pos] +
                      f'<span class="hl {cls}" data-cidx="{cidx}" onclick="onHlClickByChunk({cidx})">{badge}' +
                      result_html[pos:tag_end] +
                      f'</span>' +
                      result_html[tag_end:])

    return result_html

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
            # Convert plain text thành HTML paragraphs
            paras = [p.strip() for p in main_text.split('\n\n') if p.strip()]
            main_html = ''.join(f'<p>{p.replace(chr(10), "<br>")}</p>' for p in paras) if paras else f'<p>{main_text}</p>'
        else:
            main_file = request.files.get('main_file')
            if not main_file:
                return jsonify({"error": "Thiếu file tài liệu chính"}), 400
            result_main = extract_text_from_file(main_file, return_html=True)
            main_text, main_html = result_main if isinstance(result_main, tuple) else (result_main, None)
            if len(main_text) < 50:
                return jsonify({"error": "Tài liệu quá ngắn hoặc không đọc được"}), 400

        # File mẫu là tùy chọn
        template_file = request.files.get('template_file')
        if template_file and template_file.filename:
            result_tmpl = extract_text_from_file(template_file, return_html=False)
            template_text = result_tmpl if isinstance(result_tmpl, str) else result_tmpl[0]
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
                sources = search_google_free_fetch(query_text)
            r['sources'] = sources

        avg_ai = round(sum(r['ai_score'] for r in results) / total_chunks)
        avg_plag = round(sum(r['plag_score'] for r in results) / total_chunks)

        # Tạo document view có highlight (dùng chunk không overlap để map chính xác)
        no_overlap_chunks = chunk_text_no_overlap(cleaned_text, chunk_size=1500)
        doc_blocks = build_highlighted_document(cleaned_text, results, no_overlap_chunks)

        # Build highlighted HTML document từ main_html (giữ định dạng)
        html_blocks = build_html_doc_blocks(main_html, cleaned_text, results)

        return jsonify({
            "success": True,
            "html_content": main_html,  # HTML gốc từ Word
            "html_blocks": html_blocks,  # HTML đã highlight
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


@app.route('/rewrite', methods=['POST'])
def rewrite():
    """Nhận đoạn văn nghi AI/đạo văn, trả về phiên bản viết lại tự nhiên hơn."""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        issue_type = data.get('issue_type', 'ai')  # 'ai' hoặc 'plag'
        api_key = data.get('api_key', '').strip()
        context = data.get('context', '')  # đoạn xung quanh để giữ mạch văn
        style_hint = data.get('style_hint', '')  # gợi ý phong cách nếu có

        if not text:
            return jsonify({"error": "Thiếu văn bản cần sửa"}), 400
        if not api_key:
            return jsonify({"error": "Cần Anthropic API key để sử dụng tính năng này"}), 400

        client = anthropic.Anthropic(api_key=api_key)

        if issue_type == 'ai':
            system_prompt = """Bạn là chuyên gia viết lại văn bản theo phong cách người thật, tránh bị phát hiện là AI.

NHIỆM VỤ: Viết lại đoạn văn sao cho:
1. Giữ nguyên NỘI DUNG và Ý NGHĨA gốc
2. Thêm GIỌNG CÁ NHÂN: dùng "tôi", "theo tôi", "từ kinh nghiệm", "tôi nhận thấy"
3. Thêm SỰ KHÔNG HOÀN HẢO TỰ NHIÊN: câu ngắn xen câu dài (burstiness cao)
4. TRÁNH: "bên cạnh đó", "ngoài ra", "tóm lại", "nhìn chung", "thứ nhất/hai/ba"
5. THÊM: từ nối thông thường, câu hỏi tu từ, cảm xúc nhẹ, ví dụ cụ thể
6. Dùng từ ngữ BÌNH DÂN hơn, tránh từ hàn lâm đồng đều

Trả về JSON:
{
  "rewritten": "đoạn văn đã viết lại",
  "changes": ["thay đổi 1", "thay đổi 2", "thay đổi 3"],
  "tips": "gợi ý thêm nếu cần"
}"""
        else:  # plag
            system_prompt = """Bạn là chuyên gia viết lại văn bản để tránh đạo văn.

NHIỆM VỤ: Viết lại đoạn văn sao cho:
1. Giữ nguyên Ý NGHĨA và THÔNG TIN gốc
2. Thay đổi HOÀN TOÀN cấu trúc câu và từ ngữ
3. Diễn đạt theo cách KHÁC BIỆT: thay đổi thứ tự, dùng từ đồng nghĩa, ghép/tách câu
4. Thêm góc nhìn cá nhân hoặc ví dụ thực tế
5. Đảm bảo văn phong tự nhiên, không máy móc

Trả về JSON:
{
  "rewritten": "đoạn văn đã viết lại",
  "changes": ["thay đổi 1", "thay đổi 2", "thay đổi 3"],
  "tips": "gợi ý thêm nếu cần"
}"""

        user_msg = f"Viết lại đoạn văn này:\n\n{text[:1500]}"
        if context:
            user_msg += f"\n\nNgữ cảnh xung quanh (để giữ mạch văn):\n{context[:300]}"
        if style_hint:
            user_msg += f"\n\nYêu cầu thêm: {style_hint}"

        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            system=system_prompt,
            messages=[{"role": "user", "content": user_msg}]
        )

        raw = msg.content[0].text
        clean = raw.replace('```json','').replace('```','').strip()
        result = json.loads(clean)

        # Tính điểm AI của bản viết lại để so sánh
        new_score, _ = heuristic_ai_score(result.get('rewritten', ''))

        return jsonify({
            "success": True,
            "rewritten": result.get("rewritten", ""),
            "changes": result.get("changes", []),
            "tips": result.get("tips", ""),
            "original_length": len(text),
            "new_ai_score": new_score
        })

    except json.JSONDecodeError:
        # Nếu Claude không trả về JSON đúng, lấy text thô
        raw = msg.content[0].text if 'msg' in locals() else ''
        return jsonify({
            "success": True,
            "rewritten": raw,
            "changes": [],
            "tips": "",
            "new_ai_score": 0
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# ─── Free Rewrite Engine FINAL ──────────────────────────────────────────────

import random as _rnd, re as _re

def rewrite_text_free(text, issue_type='ai', style_hint=''):
    """
    Viết lại văn bản — FINAL version.
    Nguyên tắc: ít bước, mỗi bước làm đúng 1 việc, không chồng chéo.
    """
    _rnd.seed()
    result = text
    changes = []

    # ── BƯỚC 1: Phân câu ──────────────────────────────────────────────────────
    def split_sents(t):
        return [s.strip() for s in _re.split(r'(?<=[.!?])\s+', t.strip()) if s.strip()]

    def join_sents(sents):
        return ' '.join(sents)

    if issue_type not in ('ai', 'both', 'plag'):
        issue_type = 'ai'

    # ── BƯỚC 2: Ghép câu thiếu chủ ngữ VÀO CÂU TRƯỚC ────────────────────────
    # Câu thiếu chủ ngữ: bắt đầu bằng từ nối + động từ thường (chữ thường ngay sau)
    SUBJECTLESS = ['Đồng thời', 'Từ đó', 'Nhờ đó', 'Qua đó', 'Từ đây',
                   'Do đó vậy', 'Cùng lúc đó', 'Song song']
    sents = split_sents(result)
    merged = []
    for i, s in enumerate(sents):
        fused = False
        for conn in SUBJECTLESS:
            if s.startswith(conn):
                after = s[len(conn):].lstrip(', ')
                if after and after[0].islower() and merged:
                    # Ghép: loại bỏ dấu chấm cuối câu trước, thêm dấu phẩy
                    prev = merged[-1].rstrip('.!?')
                    merged[-1] = prev + ', ' + conn.lower() + ' ' + after
                    fused = True
                    changes.append('Ghép câu thiếu chủ ngữ vào câu trước')
                    break
        if not fused:
            merged.append(s)
    result = join_sents(merged)

    if issue_type in ('ai', 'both'):
        # ── BƯỚC 3: Xử lý TỪng câu — thay connector đầu câu ──────────────────
        # Map: (từ cần thay, danh sách thay thế)
        # Chú ý: thay ĐẦU CÂU, giữ phần còn lại nguyên
        CONNECTOR_SUBS = [
            ('Bên cạnh đó',          ['Thêm vào đó,', 'Ngoài việc này,']),
            ('Ngoài ra',             ['Thêm nữa,', 'Và cũng,']),
            ('Hơn nữa',              ['Không chỉ vậy,', 'Và,']),
            ('Tóm lại',              ['Nhìn lại,', 'Có thể thấy,']),
            ('Nhìn chung',           ['Thực ra,', 'Nói thật,']),
            ('Quan trọng là',        ['Điều cần nói,']),
            ('Cần lưu ý rằng',       ['Đáng chú ý,']),
            ('Đặc biệt là',          ['Nhất là,', 'Cụ thể,']),
            ('Theo đó',              ['Vì vậy,', 'Do đó,']),
            ('Không thể phủ nhận',   ['Rõ ràng,']),
        ]
        # Thay liệt kê
        NUMBERED_SUBS = [
            ('Thứ nhất',  ['Trước hết,', 'Đầu tiên,']),
            ('Thứ hai',   ['Tiếp đến,',  'Kế đó,']),
            ('Thứ ba',    ['Ngoài ra,',  'Cuối cùng,']),
            ('Thứ tư',    ['Thêm nữa,']),
            ('Thứ năm',   ['Và,']),
        ]

        sents = split_sents(result)
        new_sents = []
        for s in sents:
            new_s = s
            # Thay liệt kê trước
            for orig, opts in NUMBERED_SUBS:
                pat = _re.compile(r'^' + _re.escape(orig) + r'[,:]?\s+', _re.IGNORECASE)
                if pat.match(new_s):
                    repl = _rnd.choice(opts)
                    rest = pat.sub('', new_s)
                    new_s = repl + ' ' + rest[0].lower() + rest[1:] if rest else repl
                    changes.append(f'Liệt kê: {orig} → {repl}')
                    break
            # Thay connector đầu câu (chỉ nếu chưa thay ở trên)
            if new_s == s:
                for orig, opts in CONNECTOR_SUBS:
                    pat = _re.compile(r'^' + _re.escape(orig) + r'[,:]?\s+', _re.IGNORECASE)
                    if pat.match(new_s):
                        repl = _rnd.choice(opts)
                        rest = pat.sub('', new_s)
                        new_s = repl + ' ' + rest[0].lower() + rest[1:] if rest else repl
                        changes.append(f'Connector: {orig} → {repl}')
                        break
            new_sents.append(new_s)
        result = join_sents(new_sents)

        # ── BƯỚC 4: Thay connector giữa câu ───────────────────────────────────
        MID_SUBS = [
            (r',\s*cần lưu ý rằng\s+',  ', đáng chú ý là '),
            (r',\s*cần lưu ý\s+',        ', cần biết '),
            (r'\s+cần lưu ý rằng\s+',    ' cần biết rằng '),
            (r',\s*quan trọng là\s+',    ', điều cần nhớ là '),
            (r',\s*bên cạnh đó,?\s+',    ', và '),
            (r',\s*ngoài ra,?\s+',        ', thêm nữa '),
            (r',\s*hơn nữa,?\s+',         ', không chỉ vậy '),
            (r'(?<=\w) cần lưu ý rằng\s+', ' — đáng chú ý là '),
        ]
        for pat, repl in MID_SUBS:
            new_r = _re.sub(pat, repl, result, flags=_re.IGNORECASE)
            if new_r != result:
                changes.append('Thay connector giữa câu')
                result = new_r

        # ── BƯỚC 5: Thay từ nội dung (tối đa 5 từ) ───────────────────────────
        WORD_SUBS = [
            (r'\bphương pháp\b',    ['cách', 'cách làm']),
            (r'\bgiải pháp\b',      ['cách giải quyết', 'phương án']),
            (r'\bquan trọng\b',     ['thiết yếu', 'then chốt']),
            (r'\bbền vững\b',       ['lâu dài', 'ổn định']),
            (r'\bphù hợp\b',        ['đúng', 'hợp lý']),
            (r'\bđáng kể\b',        ['rõ rệt', 'rõ ràng']),
            (r'\btriển khai\b',     ['áp dụng', 'tiến hành']),
            (r'\bthực hiện\b',      ['làm', 'tiến hành']),
            (r'\bkết quả\b',        ['thành quả', 'điều đạt được']),
            (r'\bnăng lực\b',       ['khả năng', 'kỹ năng']),
            (r'\bmục tiêu\b',       ['mục đích', 'điều muốn đạt']),
            (r'\btoàn diện\b',      ['đầy đủ', 'đồng bộ']),
            (r'\btích cực\b',       ['chủ động', 'tốt']),
            (r'\bhiệu quả\b',       ['tốt', 'có ích']),
            (r'\bnâng cao\b',       ['cải thiện', 'tăng']),
            (r'\bchất lượng\b',     ['trình độ', 'thực chất']),
            (r'\bmôi trường\b',     ['không gian', 'bầu không khí']),
            (r'\bhình thành\b',     ['tạo ra', 'xây dựng']),
            (r'\bgóp phần\b',       ['giúp', 'hỗ trợ']),
            # đóng vai trò: KHÔNG thay vì dễ tạo câu sai ngữ pháp
        ]
        n_word_replaced = 0
        for pat, opts in WORD_SUBS:
            if n_word_replaced >= 5: break
            rx = _re.compile(pat, _re.IGNORECASE)
            m = rx.search(result)
            if m:
                repl = _rnd.choice(opts)
                orig = m.group(0)
                if orig[0].isupper(): repl = repl[0].upper() + repl[1:]
                result = result[:m.start()] + repl + result[m.end():]
                changes.append(f'Từ: {orig} → {repl}')
                n_word_replaced += 1

        # ── BƯỚC 6: Thêm giọng giáo viên ─────────────────────────────────────
        TEACHER = [
            'Từ thực tế giảng dạy, tôi thấy ',
            'Qua kinh nghiệm đứng lớp, tôi nhận ra ',
            'Trong quá trình dạy, ',
            'Tôi nhận thấy rằng ',
            'Thực tế cho thấy ',
        ]
        has_voice = any(p in result.lower() for p in ['tôi ', 'mình ', 'từ kinh nghiệm', 'từ thực tế'])
        if not has_voice:
            sents2 = split_sents(result)
            SKIP_S = ['thêm', 'tiếp', 'kế đó', 'trước hết', 'đầu tiên', 'ngoài ra',
                      'cuối cùng', 'và,', 'nhìn lại', 'có thể', 'thực ra', 'thực tế']
            SKIP_C = ['thứ nhất', 'thứ hai', 'một là', 'hai là']
            target = -1
            for idx in range(1, min(4, len(sents2))):
                sl = sents2[idx].lower()
                if (not any(sl.startswith(x) for x in SKIP_S)
                        and not any(x in sl for x in SKIP_C)
                        and len(sents2[idx].split()) >= 5):
                    target = idx
                    break
            if target >= 0:
                starter = _rnd.choice(TEACHER)
                s2 = sents2[target]
                sents2[target] = starter + s2[0].lower() + s2[1:]
                result = join_sents(sents2)
                changes.append('Thêm giọng cá nhân')

    if issue_type in ('plag', 'both'):
        # Đảo câu giữa
        sents3 = split_sents(result)
        if len(sents3) >= 5:
            m3 = len(sents3) // 2
            sents3[m3-1], sents3[m3] = sents3[m3], sents3[m3-1]
            result = join_sents(sents3)
            changes.append('Đổi thứ tự câu')

    # ── BƯỚC CUỐI: Dọn dẹp ────────────────────────────────────────────────────
    # Dọn connector kép: "Và, đáng chú ý" → "Đáng chú ý"
    result = _re.sub(r'^Và,\s+(đáng|nên|điều|cần|thực)', lambda m: m.group(1).capitalize(), result, flags=_re.MULTILINE|_re.IGNORECASE)
    result = _re.sub(r'(?<=\. )Và,\s+(đáng|nên|điều|cần|thực)', lambda m: m.group(1).capitalize(), result, flags=_re.IGNORECASE)
    # Từ lặp liền kề: "tốt, tốt" → "tốt"
    result = _re.sub(r'\b(\w{3,}),\s+\1\b', r'\1', result)
    result = _re.sub(r'\b(\w{2,})\s+\1\b', r'\1', result)   # từ lặp
    result = _re.sub(r',\s*,+', ',', result)                  # phẩy kép
    result = _re.sub(r'  +', ' ', result).strip()             # khoảng trắng
    result = _re.sub(r'(?<=\. )([a-zàáâãèéêìíòóôõùúýăđơư])',
                     lambda m: m.group(1).upper(), result)    # hoa đầu câu
    result = _re.sub(r',\s*\.', '.', result)                   # phẩy trước chấm
    if result and result[-1] not in '.!?': result += '.'

    tips = ''
    if style_hint:
        sl = style_hint.lower()
        if any(k in sl for k in ['ví dụ', 'cụ thể']):
            tips = 'Thêm ví dụ: "Khi tôi dạy lớp 8A..." sẽ rất thuyết phục.'
        elif 'ngắn' in sl:
            tips = 'Câu ngắn 10-15 từ giảm điểm AI hiệu quả nhất.'
        elif any(k in sl for k in ['trang trọng', 'hành chính']):
            tips = 'Thêm số liệu cụ thể (%, năm học, tên lớp) để tăng tính xác thực.'

    seen, unique = set(), []
    for c in changes:
        if c not in seen: seen.add(c); unique.append(c)

    return {'rewritten': result, 'changes': unique[:8], 'tips': tips}



@app.route('/rewrite_free', methods=['POST'])
def rewrite_free_route():
    """Viết lại miễn phí — không cần API key."""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        issue_type = data.get('issue_type', 'ai')
        style_hint = data.get('style_hint', '')
        if not text:
            return jsonify({"error": "Thiếu văn bản"}), 400
        result = rewrite_text_free(text, issue_type, style_hint)
        new_score, _ = heuristic_ai_score(result['rewritten'])
        return jsonify({
            "success": True,
            "rewritten": result['rewritten'],
            "changes": result['changes'],
            "tips": result.get('tips', ''),
            "new_ai_score": new_score
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
