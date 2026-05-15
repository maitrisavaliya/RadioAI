"""
Minimal hero banner — clean, single accent, no complex animations.
"""

HERO_HTML = (
    '<style>'
    '@import url("https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap");'
    '.rai-hero{'
        'background:#1e293b;border:1px solid #334155;border-radius:12px;'
        'padding:28px 32px;margin-bottom:24px;'
        'display:flex;align-items:center;justify-content:space-between;'
    '}'
    '.rai-hero-left{max-width:70%;}'
    '.rai-hero-label{'
        'font-family:Inter,sans-serif;font-size:0.75rem;font-weight:500;'
        'color:#94a3b8;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:8px;'
    '}'
    '.rai-hero-title{'
        'font-family:Inter,sans-serif;font-size:2rem;font-weight:600;'
        'color:#e2e8f0;letter-spacing:-0.02em;line-height:1.15;margin-bottom:6px;'
    '}'
    '.rai-hero-title span{color:#3b82f6;}'
    '.rai-hero-sub{'
        'font-family:Inter,sans-serif;font-size:0.875rem;color:#94a3b8;'
        'margin-bottom:16px;'
    '}'
    '.rai-badge{'
        'display:inline-block;background:#0f172a;border:1px solid #334155;'
        'color:#94a3b8;border-radius:6px;padding:3px 10px;'
        'font-family:Inter,sans-serif;font-size:0.75rem;font-weight:500;'
        'margin:2px;'
    '}'
    '.rai-hero-right{'
        'font-size:3rem;opacity:0.15;'
    '}'
    '</style>'
    '<div class="rai-hero">'
        '<div class="rai-hero-left">'
            '<div class="rai-hero-label">Medical Imaging · Research Demo</div>'
            '<div class="rai-hero-title">Radio<span>AI</span></div>'
            '<div class="rai-hero-sub">'
                'Upload a scan. The system identifies the type and analyses it automatically.'
            '</div>'
            '<div>'
                '<span class="rai-badge">🫁 CT Scan</span>'
                '<span class="rai-badge">🔬 Ultrasound</span>'
                '<span class="rai-badge">🧠 MRI</span>'
            '</div>'
        '</div>'
        '<div class="rai-hero-right">🩺</div>'
    '</div>'
)

UPLOAD_ANIMATION_CSS = """
<style>
@keyframes fadeUp {
    from { opacity:0; transform:translateY(12px); }
    to   { opacity:1; transform:translateY(0); }
}
.fade-up { animation: fadeUp 0.25s ease-out both; }
</style>
"""

RESULT_ANIMATION = ""
