#!/usr/bin/env python3
"""
build_arch.py — 从拆分后的 Markdown 生成 HTML，替换 web/index.html 中的 NV 微架构导航和 section。
优先读取形如 0.xxx.md 的拆分文件；若不存在，则回退到 NV微架构梳理.md 单文件模式。
用法：python3 user/build_arch.py
"""

import re
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
ROOT = SCRIPT_DIR.parent
LEGACY_MD_FILE = SCRIPT_DIR / 'NV微架构梳理.md'
HTML_FILE = ROOT / 'web' / 'index.html'
SRC_IMAGES = SCRIPT_DIR / 'images'
DST_IMAGES = ROOT / 'web' / 'images'

SPLIT_MD_RE = re.compile(r'^(?P<order>\d+)\.(?P<slug>.+)\.md$')
NAV_START_MARKER = '<!-- ARCH_NAV_START -->'
NAV_END_MARKER = '<!-- ARCH_NAV_END -->'
SECTIONS_START_MARKER = '<!-- ARCH_SECTIONS_START -->'
SECTIONS_END_MARKER = '<!-- ARCH_SECTIONS_END -->'

# ── Markdown 解析 ──────────────────────────────────────────

def parse_md(text):
    """解析整个 md 文件，返回 (architectures, compare_table_md, compare_title)"""
    # 提取对比表标题
    title_match = re.search(r'^## (.+)', text, flags=re.MULTILINE)
    compare_title = title_match.group(1).strip() if title_match else '微架构演进对比'
    # 分离对比表
    compare_split = re.split(r'^## .+\s*$', text, maxsplit=1, flags=re.MULTILINE)
    if len(compare_split) > 1:
        # 表格在前面：compare_split[0] 为空或空白，[1] 包含表格 + 架构内容
        # 表格在后面：compare_split[0] 包含架构内容，[1] 包含表格
        before = compare_split[0].strip()
        after = compare_split[1].strip()
        if not before:
            # 表格在最前面，架构内容在后面
            # 从 after 中分离表格和架构块（第一个 --- 开头的 frontmatter）
            fm_match = re.search(r'\n---\n\n---\nid:', after)
            if fm_match:
                compare_md = after[:fm_match.start()].strip()
                arch_text = after[fm_match.start():].strip()
            else:
                compare_md = after
                arch_text = ''
        else:
            # 表格在末尾（原有逻辑）
            arch_text = before
            compare_md = after
    else:
        arch_text = text
        compare_md = ''

    # 找到所有 frontmatter 块：每个以 ---\nid: ... 开头
    # 用正则找到每个 --- 开头的 frontmatter 位置
    arch_blocks = re.split(r'\n---\n\n---\n', arch_text)
    # 第一个块以文件开头的 --- 开始
    architectures = []
    for block in arch_blocks:
        block = block.strip()
        if not block:
            continue
        # 确保块以 --- 开头（第一个块自带，后面的被 split 去掉了）
        if not block.startswith('---'):
            block = '---\n' + block
        arch = parse_arch_block(block)
        if arch:
            architectures.append(arch)

    return architectures, compare_md, compare_title


def slugify(text):
    """将任意文本转为适合 HTML id 的 slug。"""
    slug = re.sub(r'[^a-zA-Z0-9]+', '-', text).strip('-').lower()
    return slug or 'section'


def humanize_slug(slug):
    """将文件名中的 slug 转为导航显示文案。"""
    parts = [part for part in re.split(r'[-_]+', slug) if part]
    if not parts:
        return 'NV 微架构'
    return '-'.join(part[:1].upper() + part[1:] for part in parts)


def build_page_label(slug, archs):
    """根据拆分文件内容生成导航/页面标题。"""
    if slug == 'table':
        return '微架构发展与比较'
    if len(archs) == 1:
        return archs[0]['name']
    return humanize_slug(slug)


def make_page(source, order, slug, nav_label, section_title, section_id, icon, archs, compare_md, compare_title):
    """构造单个页面配置。"""
    return {
        'source': source,
        'order': order,
        'slug': slug,
        'nav_label': nav_label,
        'section_title': section_title,
        'section_id': section_id,
        'icon': icon,
        'archs': archs,
        'compare_md': compare_md,
        'compare_title': compare_title,
    }


def expand_source_to_pages(source, order, slug, archs, compare_md, compare_title):
    """将一个 Markdown 源展开为一个或多个页面。"""
    pages = []

    if compare_md:
        compare_slug = 'table' if slug == 'overview' else slug
        pages.append(make_page(
            source=source,
            order=order,
            slug=compare_slug,
            nav_label='微架构发展与比较',
            section_title='微架构发展与比较',
            section_id=f'nv-arch-{order}-{slugify(compare_slug)}',
            icon='📊',
            archs=[],
            compare_md=compare_md,
            compare_title=compare_title,
        ))

    if archs:
        if len(archs) == 1:
            arch = archs[0]
            page_slug = slug if slug != 'overview' else slugify(arch['id'])
            pages.append(make_page(
                source=source,
                order=order,
                slug=page_slug,
                nav_label=arch['name'],
                section_title=arch['name'],
                section_id=f'nv-arch-{order}-{slugify(page_slug)}',
                icon='🏗️',
                archs=[arch],
                compare_md='',
                compare_title='',
            ))
        else:
            for arch in archs:
                page_slug = f'{slug}-{slugify(arch["id"])}'
                pages.append(make_page(
                    source=source,
                    order=order,
                    slug=page_slug,
                    nav_label=arch['name'],
                    section_title=arch['name'],
                    section_id=f'nv-arch-{order}-{slugify(page_slug)}',
                    icon='🏗️',
                    archs=[arch],
                    compare_md='',
                    compare_title='',
                ))

    return pages


def discover_pages():
    """发现需要生成的页面，优先使用拆分 md。"""
    split_files = []
    for path in SCRIPT_DIR.glob('*.md'):
        if path.name == LEGACY_MD_FILE.name:
            continue
        match = SPLIT_MD_RE.match(path.name)
        if not match:
            continue
        split_files.append((int(match.group('order')), match.group('slug'), path))

    if split_files:
        split_files.sort(key=lambda item: (item[0], item[1].lower()))
        pages = []
        for order, slug, path in split_files:
            md_text = path.read_text(encoding='utf-8')
            archs, compare_md, compare_title = parse_md(md_text)
            pages.extend(expand_source_to_pages(path, order, slug, archs, compare_md, compare_title))
        return pages

    if not LEGACY_MD_FILE.exists():
        raise RuntimeError('找不到拆分 Markdown，也找不到 user/NV微架构梳理.md')

    md_text = LEGACY_MD_FILE.read_text(encoding='utf-8')
    archs, compare_md, compare_title = parse_md(md_text)
    return expand_source_to_pages(LEGACY_MD_FILE, 0, 'overview', archs, compare_md, compare_title)


def parse_arch_block(block):
    """解析单个架构块：frontmatter + sections"""
    # 去掉末尾可能残留的 ---
    block = block.rstrip()
    if block.endswith('\n---'):
        block = block[:-4]

    # 提取 frontmatter
    fm_match = re.match(r'^---\s*\n(.*?)\n---\s*\n?(.*)', block, re.DOTALL)
    if not fm_match:
        return None

    meta_text = fm_match.group(1)
    content = fm_match.group(2)

    meta = {}
    for line in meta_text.strip().split('\n'):
        m = re.match(r'^(\w[\w_]*)\s*:\s*(.+)$', line)
        if m:
            meta[m.group(1)] = m.group(2).strip()

    if 'id' not in meta:
        return None

    # 按 ### 分段
    sections = []
    parts = re.split(r'^### (.+)$', content, flags=re.MULTILINE)
    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        body = parts[i + 1] if i + 1 < len(parts) else ''
        sections.append({'title': title, 'body': body.strip()})

    return {**meta, 'sections': sections}


# ── Markdown → HTML 转换 ────────────────────────────────────

def inline_md(text):
    """行内 markdown：**粗体**, `code`, [链接](url), 裸 URL"""
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    text = re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)', r'<a href="\2" target="_blank">\1</a>', text)
    text = re.sub(r'(?<!["\'>])(https?://[^\s<,，。）\)]+)', r'<a href="\1" target="_blank">\1</a>', text)
    return text


def parse_sm_config(body):
    """解析 SM 配置段为列表 [{label, value, highlight}]"""
    items = []
    for line in body.split('\n'):
        m = re.match(r'^-\s+(.+?):\s*(.+)$', line)
        if m:
            highlight = '[highlight]' in m.group(2)
            value = m.group(2).replace('[highlight]', '').strip()
            items.append({'label': m.group(1).strip(), 'value': value, 'highlight': highlight})
    return items


def notes_to_html(body):
    """将说明段的 markdown 转为 HTML"""
    lines = body.split('\n')
    html_parts = []
    in_list = False
    notes_to_html._in_sublist = False
    i = 0

    while i < len(lines):
        line = lines[i]

        # 双图: [images: a.png | b.png]
        img2 = re.match(r'^\[images:\s*(.+?)\s*\|\s*(.+?)\s*\]$', line)
        if img2:
            if in_list:
                html_parts.append('</ul>')
                in_list = False
            cap_left, cap_right = '', ''
            if i + 1 < len(lines):
                cap_m = re.match(r'^\[captions:\s*(.+?)\s*\|\s*(.+?)\s*\]$', lines[i + 1])
                if cap_m:
                    cap_left, cap_right = cap_m.group(1).strip(), cap_m.group(2).strip()
                    i += 1
            html_parts.append(
                f'<div class="arch-content-row">'
                f'<div class="arch-figure-half">'
                f'<img src="images/{img2.group(1).strip()}" alt="{cap_left}">'
                f'{f"<div class=\"paper-figure-caption\">{cap_left}</div>" if cap_left else ""}'
                f'</div>'
                f'<div class="arch-figure-half">'
                f'<img src="images/{img2.group(2).strip()}" alt="{cap_right}">'
                f'{f"<div class=\"paper-figure-caption\">{cap_right}</div>" if cap_right else ""}'
                f'</div></div>'
            )
            i += 1
            continue

        # 单图: [image: a.png] 或 [full-image: a.png] (全宽显示)
        img1 = re.match(r'^\[image:\s*(.+?)\s*\]$', line)
        full_img = re.match(r'^\[full-image:\s*(.+?)\s*\]$', line)

        if img1:
            if in_list:
                html_parts.append('</ul>')
                in_list = False
            caption = ''
            if i + 1 < len(lines):
                cap_m = re.match(r'^\[caption:\s*(.+?)\s*\]$', lines[i + 1])
                if cap_m:
                    caption = cap_m.group(1).strip()
                    i += 1
            html_parts.append(
                f'<div class="paper-figure" style="margin-top:12px">'
                f'<img src="images/{img1.group(1).strip()}" alt="{caption}" style="max-width:100%; height:auto;">'
                f'{f"<div class=\"paper-figure-caption\">{caption}</div>" if caption else ""}'
                f'</div>'
            )
            i += 1
            continue
        elif full_img:
            # 全宽图片
            if in_list:
                html_parts.append('</ul>')
                in_list = False
            caption = ''
            if i + 1 < len(lines):
                cap_m = re.match(r'^\[caption:\s*(.+?)\s*\]$', lines[i + 1])
                if cap_m:
                    caption = cap_m.group(1).strip()
                    i += 1
            html_parts.append(
                f'<div class="full-width-figure">'
                f'<img src="images/{full_img.group(1).strip()}" alt="{caption}" style="max-width:100%; height:auto;">'
                f'{f"<div class=\"paper-figure-caption\">{caption}</div>" if caption else ""}'
                f'</div>'
            )
            i += 1
            continue

        # 跳过独立的 caption 行（已被上面消费）
        if re.match(r'^\[captions?:', line):
            i += 1
            continue

        # 列表项（无序 - 或有序 1.），支持一级和二级（2+空格缩进）
        ul_m = re.match(r'^-\s+(.+)$', line)
        ol_m = re.match(r'^\d+\.\s+(.+)$', line)
        sub_ul_m = re.match(r'^ {2,}-\s+(.+)$', line)
        sub_ol_m = re.match(r'^ {2,}\d+\.\s+(.+)$', line)

        if sub_ul_m or sub_ol_m:
            # 二级列表项
            content = (sub_ul_m or sub_ol_m).group(1)
            tag = 'ul' if sub_ul_m else 'ol'
            if not getattr(notes_to_html, '_in_sublist', False):
                html_parts.append(f'<{tag} class="arch-sublist">')
                notes_to_html._in_sublist = tag
            elif notes_to_html._in_sublist != tag:
                html_parts.append(f'</{notes_to_html._in_sublist}>')
                html_parts.append(f'<{tag} class="arch-sublist">')
                notes_to_html._in_sublist = tag
            html_parts.append(f'<li>{inline_md(content)}</li>')
            i += 1
            continue
        else:
            if getattr(notes_to_html, '_in_sublist', False):
                html_parts.append(f'</{notes_to_html._in_sublist}>')
                notes_to_html._in_sublist = False

        if ul_m or ol_m:
            content = (ul_m or ol_m).group(1)
            new_type = 'ul' if ul_m else 'ol'
            if in_list and in_list != new_type:
                html_parts.append(f'</{in_list}>')
                in_list = False
            if not in_list:
                cls = ' class="arch-list"' if new_type == 'ul' else ' class="arch-list-ol"'
                html_parts.append(f'<{new_type}{cls}>')
                in_list = new_type
            html_parts.append(f'<li>{inline_md(content)}</li>')
            i += 1
            continue

        # 空行
        if line.strip() == '':
            if getattr(notes_to_html, '_in_sublist', False):
                html_parts.append(f'</{notes_to_html._in_sublist}>')
                notes_to_html._in_sublist = False
            if in_list:
                html_parts.append(f'</{in_list}>')
                in_list = False
            i += 1
            continue

        # 普通段落
        if getattr(notes_to_html, '_in_sublist', False):
            html_parts.append(f'</{notes_to_html._in_sublist}>')
            notes_to_html._in_sublist = False
        if in_list:
            html_parts.append(f'</{in_list}>')
            in_list = False
        html_parts.append(f'<p>{inline_md(line)}</p>')
        i += 1

    if getattr(notes_to_html, '_in_sublist', False):
        html_parts.append(f'</{notes_to_html._in_sublist}>')
        notes_to_html._in_sublist = False
    if in_list:
        html_parts.append(f'</{in_list}>')

    return '\n                                    '.join(html_parts)


# ── HTML 生成 ──────────────────────────────────────────────

def render_sm_grid(items):
    """生成 SM 配置 grid HTML"""
    parts = []
    for item in items:
        cls = ' highlight' if item['highlight'] else ''
        parts.append(
            f'<div class="arch-sm-item{cls}">'
            f'<span class="sm-value">{item["value"]}</span>'
            f'<span class="sm-label">{item["label"]}</span>'
            f'</div>'
        )
    return '\n                                    '.join(parts)


def render_figure(arch):
    """生成图片或占位符 HTML"""
    if 'image' in arch:
        caption = arch.get('image_caption', '')
        html = f'<div class="arch-figure">\n'
        html += f'                                <img src="images/{arch["image"]}" alt="{arch["name"]} 架构图">\n'
        if caption:
            html += f'                                <div class="paper-figure-caption">{caption}</div>\n'
        html += f'                            </div>'
        return html
    else:
        return (f'<div class="arch-figure">\n'
                f'                                <div class="arch-figure-placeholder">{arch["name"]} SM 架构图<br>(待补充)</div>\n'
                f'                            </div>')


def render_extra_section(body):
    """渲染额外子段落（如 Volta SIMT、Ampere A100 对比）"""
    lines = body.split('\n')
    html = ''
    i = 0
    text_lines = []

    def flush_text():
        nonlocal html
        if text_lines:
            text_html = notes_to_html('\n'.join(text_lines))
            html += f'                        <div class="arch-notes" style="margin-top:12px">\n'
            html += f'                            {text_html}\n'
            html += f'                        </div>\n'
            text_lines.clear()

    while i < len(lines):
        line = lines[i]

        # 双图
        img2 = re.match(r'^\[images:\s*(.+?)\s*\|\s*(.+?)\s*\]$', line)
        if img2:
            flush_text()
            cap_left, cap_right = '', ''
            if i + 1 < len(lines):
                cap_m = re.match(r'^\[captions:\s*(.+?)\s*\|\s*(.+?)\s*\]$', lines[i + 1])
                if cap_m:
                    cap_left, cap_right = cap_m.group(1).strip(), cap_m.group(2).strip()
                    i += 1
            html += f'                        <div class="arch-content-row">\n'
            html += f'                            <div class="arch-figure-half">\n'
            html += f'                                <img src="images/{img2.group(1).strip()}" alt="{cap_left}">\n'
            if cap_left:
                html += f'                                <div class="paper-figure-caption">{cap_left}</div>\n'
            html += f'                            </div>\n'
            html += f'                            <div class="arch-figure-half">\n'
            html += f'                                <img src="images/{img2.group(2).strip()}" alt="{cap_right}">\n'
            if cap_right:
                html += f'                                <div class="paper-figure-caption">{cap_right}</div>\n'
            html += f'                            </div>\n'
            html += f'                        </div>\n'
            i += 1
            continue

        # 单图
        img1 = re.match(r'^\[image:\s*(.+?)\s*\]$', line)
        if img1:
            flush_text()
            caption = ''
            if i + 1 < len(lines):
                cap_m = re.match(r'^\[caption:\s*(.+?)\s*\]$', lines[i + 1])
                if cap_m:
                    caption = cap_m.group(1).strip()
                    i += 1
            html += f'                        <div class="paper-figure" style="margin-top:12px">\n'
            html += f'                            <img src="images/{img1.group(1).strip()}" alt="{caption}" style="max-width:700px">\n'
            if caption:
                html += f'                            <div class="paper-figure-caption">{caption}</div>\n'
            html += f'                        </div>\n'
            i += 1
            continue

        # 跳过独�� caption
        if re.match(r'^\[captions?:', line):
            i += 1
            continue

        # 普通文本行收集
        text_lines.append(line)
        i += 1

    flush_text()
    return html


def is_promotable_top_full_image(body):
    """判断一个 section 是否仅包含可提升到顶部的 full-image（_full/-full）。"""
    lines = [line.strip() for line in body.split('\n') if line.strip()]
    if not lines:
        return False
    image_match = re.match(r'^\[full-image:\s*(.+?)\s*\]$', lines[0])
    if not image_match:
        return False
    image_name = image_match.group(1).strip().lower()
    if not re.search(r'(?:_full|-full)\.[a-z0-9]+$', image_name):
        return False
    return len(lines) == 1 or (len(lines) == 2 and re.match(r'^\[caption:\s*.+?\s*\]$', lines[1]))


def render_card(arch):
    """生成单个架构卡片 HTML"""
    # Tags
    tags = [t.strip() for t in arch.get('tags', '').split(',') if t.strip()]
    tag_html = ''.join(f'\n                            <span class="paper-tag">{t}</span>' for t in tags)

    # SM config
    sm_section = next((s for s in arch['sections'] if s['title'] == 'SM 配置'), None)
    sm_items = parse_sm_config(sm_section['body']) if sm_section else []
    sm_grid = render_sm_grid(sm_items)

    # 说明 section
    notes_section = next((s for s in arch['sections'] if s['title'] == '说明'), None)
    notes_html = notes_to_html(notes_section['body']) if notes_section else ''

    # Figure
    figure_html = render_figure(arch)

    # 最前面的纯 full-image 段允许提升到卡片顶部，由 markdown 顺序驱动
    top_sections = []
    for sec in arch['sections']:
        if sec['title'] in ('SM 配置', '说明'):
            break
        if is_promotable_top_full_image(sec['body']):
            top_sections.append(sec)
        else:
            break

    top_section_ids = {id(sec) for sec in top_sections}

    # Extra sections (beyond SM 配置 and 说明)
    extra_sections = [
        s for s in arch['sections']
        if s['title'] not in ('SM 配置', '说明') and id(s) not in top_section_ids
    ]
    top_html = ''
    for sec in top_sections:
        top_html += '\n\n                        ' + render_extra_section(sec['body']).strip()

    extra_html = ''
    for sec in extra_sections:
        extra_html += f'\n\n                        <h3 class="arch-sub-title">{arch["name"]} {sec["title"]}</h3>\n'
        extra_html += render_extra_section(sec['body'])

    html = f'''                    <div class="arch-gen" id="{arch['id']}">
                        <div class="arch-gen-header">
                            <h2>{arch['name']}</h2>
                            <span class="paper-tag">{arch['year']}</span>{tag_html}
                        </div>{top_html}
                        <div class="arch-content-row">
                            <div class="arch-text">
                                <div class="arch-sm-grid">
                                    {sm_grid}
                                </div>
                                <div class="arch-notes">
                                    {notes_html}
                                </div>
                            </div>
                            {figure_html}
                        </div>{extra_html}
                    </div>'''
    return html


def render_toc(archs, compare_title=None):
    """生成目录 HTML"""
    items = []
    if compare_title:
        items.append(f'                        <li><a href="#arch-compare">{compare_title}</a></li>')
    for a in archs:
        items.append(f'                        <li><a href="#{a["id"]}">{a["name"]} ({a["year"]})</a></li>')
    return '\n'.join(items)


def render_compare_table(compare_md):
    """将 markdown 表格转为 HTML table + 后续注释 + 多个表格和标题支持"""
    lines = [l.strip() for l in compare_md.strip().split('\n') if l.strip()]
    if not lines:
        return ''

    # 按类型分组内容
    all_blocks = []
    current_block = {'type': 'unknown', 'lines': [], 'content': ''}

    i = 0
    while i < len(lines):
        line = lines[i]

        if line.startswith('## '):
            # 标题块
            if current_block['lines']:
                all_blocks.append(current_block)
            current_block = {'type': 'title', 'lines': [line], 'content': line[3:]}
        elif line.startswith('|'):
            # 表格块 - 收集所有连续的 | 行
            if current_block['type'] != 'table':
                if current_block['lines']:
                    all_blocks.append(current_block)
                current_block = {'type': 'table', 'lines': [], 'content': ''}

            # 收集连续的表格行
            while i < len(lines) and lines[i].strip().startswith('|'):
                current_block['lines'].append(lines[i])
                i += 1
            continue  # 跳过 i++，因为在循环内已处理
        elif line.startswith('>'):
            # 注释块 - 收集所有连续的 > 行
            if current_block['type'] != 'note':
                if current_block['lines']:
                    all_blocks.append(current_block)
                current_block = {'type': 'note', 'lines': [], 'content': ''}

            # 收集连续的注释行
            while i < len(lines) and lines[i].strip().startswith('>'):
                current_block['lines'].append(lines[i])
                i += 1
            continue  # 跳过 i++，因为在循环内已处理
        elif line.startswith('- **') and not line.startswith('---'):  # 避免混淆 --- 和 - **
            # 列表块 - 收集所有连续的 - ** 行
            if current_block['type'] != 'list':
                if current_block['lines']:
                    all_blocks.append(current_block)
                current_block = {'type': 'list', 'lines': [], 'content': ''}

            # 收集连续的列表行
            while i < len(lines) and lines[i].strip().startswith('- **'):
                current_block['lines'].append(lines[i])
                i += 1
            continue  # 跳过 i++，因为在循环内已处理
        elif line.startswith('### '):
            # 子标题块
            if current_block['type'] != 'subtitle':
                if current_block['lines']:
                    all_blocks.append(current_block)
                current_block = {'type': 'subtitle', 'lines': [line], 'content': line[4:]}
        elif line.startswith('- ') and not line.startswith('---') and current_block.get('type') != 'list':
            # 非列表项的 - 行作为普通文本
            if current_block['type'] != 'other':
                if current_block['lines']:
                    all_blocks.append(current_block)
                current_block = {'type': 'other', 'lines': [line], 'content': line}
            else:
                current_block['lines'].append(line)
        elif line == '---':
            # 跳过分隔符行
            pass
        else:
            # 其他行
            if current_block['type'] != 'other' and not line.startswith('---'):
                if current_block['lines']:
                    all_blocks.append(current_block)
                current_block = {'type': 'other', 'lines': [line], 'content': line}
            elif current_block['type'] == 'other':
                current_block['lines'].append(line)

        i += 1

    if current_block['lines']:
        all_blocks.append(current_block)

    html = ''
    in_container = False  # 跟踪是否在容器中

    for block in all_blocks:
        if block['type'] == 'title':
            title = block['lines'][0][3:]  # 去掉 '## '
            if not in_container:
                html += '                    <div class="matrix-container">\n'
                in_container = True
            html += f'                    <h2 class="subsection-title">{inline_md(title)}</h2>\n'
        elif block['type'] == 'table':
            # 渲染表格
            if not in_container:
                html += '                    <div class="matrix-container">\n'
                in_container = True
            headers = [c.strip() for c in block['lines'][0].split('|') if c.strip()]
            rows = []
            for line in block['lines'][2:]:  # 跳过分隔线
                cells = [c.strip() for c in line.split('|') if c.strip()]
                rows.append(cells)

            html += '<table class="hardware-matrix">\n'
            html += '                        <thead>\n                            <tr>\n'
            for h in headers:
                html += f'                                <th>{inline_md(h)}</th>\n'
            html += '                            </tr>\n                        </thead>\n'
            html += '                        <tbody>\n'
            for row in rows:
                html += '                            <tr>'
                for j, cell in enumerate(row):
                    cell_html = inline_md(cell)
                    if j == 0:
                        html += f'<td style="text-align:left;font-weight:600">{cell_html}</td>'
                    else:
                        html += f'<td>{cell_html}</td>'
                html += '</tr>\n'
            html += '                        </tbody>\n                    </table>\n'
        elif block['type'] == 'note':
            # 渲染注释
            html += '                    <div class="table-note" style="margin-top:12px;padding:12px;background:var(--bg-secondary);border-left:3px solid var(--accent-primary);border-radius:6px">\n'
            for line in block['lines']:
                if line.startswith('>'):
                    content = line[1:].strip()
                    html += f'                        <p style="margin:0;color:var(--text-secondary);font-size:13px">{inline_md(content)}</p>\n'
                else:
                    html += f'                        <p style="margin:0;color:var(--text-secondary);font-size:13px">{inline_md(line)}</p>\n'
            html += '                    </div>\n'
        elif block['type'] == 'list':
            # 渲染列表
            if not in_container:
                html += '                    <div class="matrix-container">\n'
                in_container = True
            html += '                    <div class="arch-notes" style="margin-top:24px">\n'
            for line in block['lines']:
                content = line[3:].strip()  # 去掉 '- **'
                html += f'                        <p>{inline_md(content)}</p>\n'
            html += '                    </div>\n'
        elif block['type'] == 'subtitle':
            # 渲染子标题
            subtitle = block['content']
            html += f'                    <h3>{inline_md(subtitle)}</h3>\n'
        elif block['type'] == 'other':
            # 渲染其他内容
            if not in_container:
                html += '                    <div class="matrix-container">\n'
                in_container = True
            html += '                    <div class="arch-notes" style="margin-top:12px">\n'
            for line in block['lines']:
                html += f'                        <p>{inline_md(line)}</p>\n'
            html += '                    </div>\n'

    # 关闭容器
    if in_container:
        html += '                </div>\n'

    return html


def render_section(section_title, archs, compare_md, compare_title='微架构演进对比'):
    """生成完整的 section 内容（不含 <section> 标签本身）"""
    parts = [f'                <h1 class="section-title">{section_title}</h1>']

    toc_compare_title = compare_title if compare_md else None
    if compare_md:
        table_html = render_compare_table(compare_md)
        parts.append(f'''
                <h2 class="subsection-title" id="arch-compare">{compare_title}</h2>
{table_html.rstrip()}
            ''')

    toc_count = len(archs) + (1 if toc_compare_title else 0)
    if toc_count > 1:
        toc_html = render_toc(archs, toc_compare_title)
        parts.append(f'''
                <div class="page-toc" style="margin-top:24px">
                    <h3>目录</h3>
                    <ul>
{toc_html}
                    </ul>
                </div>
            ''')

    if archs:
        cards_html = '\n\n'.join(render_card(a) for a in archs)
        parts.append(f'''
                <div class="arch-timeline" style="margin-top:32px">

{cards_html}

                </div>
            ''')

    return '\n'.join(parts)


def render_nav_items(pages):
    """生成 GPU 微架构导航列表。"""
    arch_pages = [page for page in pages if page['archs']]
    arch_total = len(arch_pages)
    arch_index = 0
    items = []
    for index, page in enumerate(pages):
        active = ' active' if index == 0 else ''
        classes = ['nav-item', 'arch-nav-item']
        if page['compare_md']:
            classes.append('arch-nav-item--compare')
            accent = 'hsl(192 88% 64%)'
            accent_soft = 'hsl(192 88% 64% / 0.22)'
        else:
            progress = arch_index / max(arch_total - 1, 1) if arch_total else 0
            hue = round(194 + (18 - 194) * progress)
            accent = f'hsl({hue} 84% 64%)'
            accent_soft = f'hsl({hue} 84% 64% / 0.22)'
            arch_index += 1

        if index == 0:
            classes.append('arch-nav-item--start')
        if index == len(pages) - 1:
            classes.append('arch-nav-item--end')

        class_attr = ' '.join(classes) + active
        style_attr = f'--nav-accent: {accent}; --nav-accent-soft: {accent_soft};'
        items.append(
            f'                    <li class="{class_attr}" data-section="{page["section_id"]}" style="{style_attr}">\n'
            f'                        <span class="icon" aria-hidden="true"></span>\n'
            f'                        <span>{page["nav_label"]}</span>\n'
            f'                    </li>'
        )
    return '\n'.join(items)


def render_sections(pages):
    """生成多个微架构 section。"""
    sections = []
    for index, page in enumerate(pages):
        active = ' active' if index == 0 else ''
        content = render_section(
            page['section_title'],
            page['archs'],
            page['compare_md'],
            page['compare_title'],
        )
        sections.append(
            f'            <section id="{page["section_id"]}" class="content-section{active}">\n'
            f'{content}\n'
            f'            </section>'
        )
    return '\n\n'.join(sections)


# ── HTML 替换 ──────────────────────────────────────────────

def replace_between_markers(html, start_marker, end_marker, new_content):
    """替换两个标记之间的内容，保留标记本身。"""
    pattern = re.compile(
        rf'({re.escape(start_marker)})(.*?)(\n[ \t]*{re.escape(end_marker)})',
        re.DOTALL,
    )
    match = pattern.search(html)
    if not match:
        raise RuntimeError(f'找不到标记: {start_marker} ... {end_marker}')

    return html[:match.end(1)] + '\n' + new_content.rstrip() + html[match.start(3):]


# ── 图片同步 ──────────────────────────────────────────────

def sync_images():
    """将 user/images/ 中的图片复制到 web/images/"""
    if not SRC_IMAGES.exists():
        return
    DST_IMAGES.mkdir(exist_ok=True)
    count = 0
    for src in SRC_IMAGES.iterdir():
        if src.is_file():
            dst = DST_IMAGES / src.name
            shutil.copy2(src, dst)
            count += 1
    print(f'  同步 {count} 张图片到 web/images/')


# ── 主流程 ────────────────────────────────────────────────

def main():
    print('发现 Markdown 页面 ...')
    pages = discover_pages()
    for page in pages:
        print(f'  - {page["source"].name} -> {page["nav_label"]}')

    total_archs = sum(len(page['archs']) for page in pages)
    print(f'  共 {len(pages)} 个页面，{total_archs} 个架构卡片')

    print('生成导航和 sections HTML ...')
    nav_html = render_nav_items(pages)
    sections_html = render_sections(pages)

    print('读取 web/index.html ...')
    html = HTML_FILE.read_text(encoding='utf-8')

    print('替换 NV 微架构导航 ...')
    new_html = replace_between_markers(html, NAV_START_MARKER, NAV_END_MARKER, nav_html)

    print('替换 NV 微架构 sections ...')
    new_html = replace_between_markers(new_html, SECTIONS_START_MARKER, SECTIONS_END_MARKER, sections_html)

    print('写回 web/index.html ...')
    HTML_FILE.write_text(new_html, encoding='utf-8')

    print('同步图片 ...')
    sync_images()

    print('完成！')


if __name__ == '__main__':
    main()
