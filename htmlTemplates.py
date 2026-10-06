css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
  overflow-x: auto;
  word-wrap: break-word;
  overflow-wrap: anywhere;
}
.chat-message .message table {
  color: #111;
}
.chat-message .message a {
  color: #9ecbff;
}
/* Always show Streamlit's sidebar collapse control (the "«" arrow). By default
   Streamlit only reveals it on sidebar hover/interaction; this forces it visible
   from first load. Selector verified for Streamlit 1.65.0. */
[data-testid="stSidebarCollapseButton"] {
  visibility: visible !important;
  opacity: 1 !important;
}
'''

import base64 as _base64
import os as _os


def _svg_avatar(bg_color, label):
    """Return an <img> tag for a self-contained SVG avatar (colored circle with a
    short label). Used as a fallback when no PNG asset is present, so the avatar
    circle is never empty. SVG is inlined as a base64 data URI (tiny, no files).
    """
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' width='78' height='78' "
        "viewBox='0 0 78 78'>"
        f"<circle cx='39' cy='39' r='39' fill='{bg_color}'/>"
        "<text x='39' y='39' font-family='Arial, sans-serif' font-size='28' "
        "font-weight='700' fill='#ffffff' text-anchor='middle' "
        f"dominant-baseline='central'>{label}</text>"
        "</svg>"
    )
    b64 = _base64.b64encode(svg.encode("utf-8")).decode()
    return f"<img src='data:image/svg+xml;base64,{b64}'>"


def _avatar_img_tag(asset_name, fallback_bg="#2e7d32", fallback_label="?"):
    """Return an <img> tag for an avatar, inlined as a base64 data URI.

    Reads the asset from the project directory (same folder as this module) and
    embeds its bytes directly so no static HTTP server is needed. If the PNG is
    missing, falls back to a self-contained SVG avatar (colored circle + label)
    so the avatar circle is always populated.
    """
    path = _os.path.join(_os.path.dirname(__file__), asset_name)
    try:
        with open(path, "rb") as f:
            b64 = _base64.b64encode(f.read()).decode()
        return f"<img src='data:image/png;base64,{b64}'>"
    except OSError:
        return _svg_avatar(fallback_bg, fallback_label)


# Avatar tags are resolved once at import time from on-disk assets (if present),
# otherwise an SVG fallback is used so the circles are never blank.
_AI_AVATAR = _avatar_img_tag("AI.png", fallback_bg="#475063", fallback_label="AI")
_USER_AVATAR = _avatar_img_tag("users.png", fallback_bg="#2b7cff", fallback_label="You")
_ICON_AVATAR = _avatar_img_tag("icon.png", fallback_bg="#475063", fallback_label="AI")

# Templates are kept on a SINGLE line with no leading whitespace. Streamlit's
# markdown renderer treats any line indented by 4+ spaces as a code block, which
# previously caused the raw HTML (e.g. "</div><div class=...") to show as literal
# text inside a grey code box. One-line strings avoid that entirely.
bot_template = (
    '<div class="chat-message bot">'
    '<div class="avatar">' + _AI_AVATAR + '</div>'
    '<div class="message">{{MSG}}</div>'
    '</div>'
)

user_template = (
    '<div class="chat-message user">'
    '<div class="avatar">' + _USER_AVATAR + '</div>'
    '<div class="message">{{MSG}}</div>'
    '</div>'
)

file_list_template = (
    '<div class="chat-message user">'
    '<div class="avatar">' + _USER_AVATAR + '</div>'
    '<div class="message">{{MSG}}</div>'
    '</div>'
)

eli_bot_template = (
    '<div class="chat-message bot">'
    '<div class="avatar">' + _ICON_AVATAR + '</div>'
    '<div class="message">{{MSG}}</div>'
    '</div>'
)

check_box = '''
   <div class="checkbox-container">
        <div id="custom-checkbox" style="width: 20px; height: 20px; border: 1px solid black; display: inline-block; cursor: pointer;" onclick="toggleCheckbox()"></div>
        <label for="custom-checkbox">Custom Div Checkbox</label>
   </div>
   <br>
    <input type="submit" value="Submit" onclick="return sendDivCheckboxValue();">
'''