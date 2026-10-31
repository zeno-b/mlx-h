"""CSS for the embedded UI: design tokens, the component kit, and the app shell.

Part of the embedded UI, split out of the single HTML_PAGE string for
readability. Assembled once at import time in ui.py, so there is no
runtime cost and no extra file I/O.

The sheet is a small design system rather than a pile of per-panel rules.
The old one styled the same control four different ways (a settings label,
a login label, a form-grid label and an inline style each had their own
look), so a button in the admin view never quite matched a button in the
chat. Now every colour, size and radius is a token, every control has ONE
implementation with all of its states (hover, active, focus-visible,
disabled, pressed), and screens are composed from those classes:

  * tokens: :root carries the light theme (the product default) together
    with html.light; html.dark and the System preference (no class on
    <html>, OS in dark mode) carry the dark theme. body.density-compact and
    body.text-small / body.text-large rescale the size tokens;
  * kit: button/.btn (+ -primary -ghost -danger -sm -icon), .input, .switch,
    .check, .range, .seg / .seg-tabs, .menu, .toast, .card, .pill, .badge,
    .table, .readonly, .kv, typography and layout utilities;
  * shell: header, chat column, composer toolbar, Settings view (rail +
    main + sticky apply bar), plus the trace, markdown, tasks, history,
    palette, folder browser, login and admin panels restyled on the tokens.

Base element rules sit inside :where() so they carry zero specificity: a
variant (.btn-primary) or a component (.seg > button) always wins over the
base without !important or selector stacking. `.hidden` stays the LAST
rule, and every id rule that sets `display` has an `#id.hidden` override in
the block just above it, because an id selector outranks `.hidden`.
"""

from __future__ import annotations

UI_CSS = r"""
   /* ================================================================ tokens */
   /* Scale tokens are theme independent. Derived colour tokens (focus ring,
      soft accent) live here too: a var() inside a custom property resolves
      on the element that declares it, which is <html> for every theme, so
      they follow html.dark / the System query without being repeated. */
   :root {
     --sp-1: 4px; --sp-2: 8px; --sp-3: 12px; --sp-4: 16px; --sp-5: 24px; --sp-6: 32px;
     --r-sm: 6px; --r-md: 10px; --r-lg: 14px; --r-pill: 999px;
     --fs-xs: 11px; --fs-sm: 12px; --fs-md: 14px; --fs-lg: 16px; --fs-xl: 20px;
     --lh: 1.5;
     --font-sans: "Raleway", "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
     --font-mono: ui-monospace, SFMono-Regular, Menlo, monospace;
     --ctl-h: 34px; --ctl-h-sm: 28px;
     --z-tip: 50; --z-pop: 60; --z-modal: 70; --z-login: 100;
     --focus-ring: 0 0 0 3px color-mix(in srgb, var(--accent) 25%, transparent);
     --accent-soft: color-mix(in srgb, var(--accent) 14%, transparent);
     --accent-hover: color-mix(in srgb, var(--accent) 88%, var(--fg));
     --header-h: 52px;
   }
   /* Light theme: the Texcel look (white, charcoal, warm orange). The brand
      accent is the readable mid-orange of the hexagon gradient. */
   :root, html.light {
     color-scheme: light;
     --bg: #f6f6f7;
     --fg: #1a1a1a;
     --muted: #5f6167;
     --accent: #e8641c;
     --accent-fg: #1a1a1a;
     --link: #b84c0f;
     --surface: #ffffff;
     --surface-2: #efeff2;
     --panel-bg: #ffffff;
     --overlay-bg: #ffffff;
     --border: #e2e2e7;
     --btn-bg: #ffffff;
     --btn-border: #d4d4db;
     --btn-hover: #f1f1f4;
     --btn-active: #e6e6eb;
     --chip-bg: #f1f1f4;
     --meter-bg: #e4e4ea;
     --knob: #ffffff;
     --code-bg: #f6f6f8;
     --code-inline-bg: #eeedf1;
     --logbox-bg: #f4f4f6;
     --logbox-fg: #2b2b30;
     --error: #c0392b;
     --warn: #b26a08;
     --success: #1f7a3d;
     --tool: #c85a12;
     --danger: #c0392b;
     --danger-fg: #ffffff;
     --scrim: rgba(20, 20, 24, 0.45);
     --shadow-1: 0 1px 2px rgba(16, 16, 20, 0.08);
     --shadow-2: 0 12px 32px rgba(16, 16, 20, 0.16);
     --tip-bg: #1a1a1a;
     --tip-fg: #f5f5f5;
     --tip-border: #3a3a3a;
     --hl-kw: #7719aa;
     --hl-str: #1f7a3d;
     --hl-num: #b26a08;
     --hl-com: #8a8a8f;
     --chart-1: #e8641c;
     --chart-2: #2563eb;
     --chart-3: #1f7a3d;
     --chart-4: #7719aa;
   }
   /* Dark theme, written twice on purpose: once for the explicit choice
      (html.dark) and once for System mode when the OS is dark. CSS has no
      way to share a declaration block between a class and a media query. */
   html.dark {
     color-scheme: dark;
     --bg: #141312;
     --fg: #ededed;
     --muted: #9b9a98;
     --accent: #f3921a;
     --accent-fg: #1a1a1a;
     --link: #f6a94e;
     --surface: #1c1b1a;
     --surface-2: #242220;
     --panel-bg: #171615;
     --overlay-bg: #201f1d;
     --border: #302e2c;
     --btn-bg: #2a2825;
     --btn-border: #4a453f;
     --btn-hover: #363330;
     --btn-active: #403c38;
     --chip-bg: #2a2825;
     --meter-bg: #2e2c29;
     --knob: #ffffff;
     --code-bg: #161514;
     --code-inline-bg: #232221;
     --logbox-bg: #0c0b0a;
     --logbox-fg: #cbc9c6;
     --error: #ff6b6b;
     --warn: #f7b015;
     --success: #9be29b;
     --tool: #f0a35a;
     --danger: #e5534b;
     --danger-fg: #ffffff;
     --scrim: rgba(0, 0, 0, 0.6);
     --shadow-1: 0 1px 2px rgba(0, 0, 0, 0.4);
     --shadow-2: 0 16px 40px rgba(0, 0, 0, 0.5);
     --tip-bg: #000000;
     --tip-fg: #ededed;
     --tip-border: #444444;
     --hl-kw: #f0a35a;
     --hl-str: #9be29b;
     --hl-num: #f7b015;
     --hl-com: #7a7a7a;
     --chart-1: #f3921a;
     --chart-2: #6ea8fe;
     --chart-3: #9be29b;
     --chart-4: #c792ea;
   }
   @media (prefers-color-scheme: dark) {
     html:not(.light):not(.dark) {
       color-scheme: dark;
       --bg: #141312;
       --fg: #ededed;
       --muted: #9b9a98;
       --accent: #f3921a;
       --accent-fg: #1a1a1a;
       --link: #f6a94e;
       --surface: #1c1b1a;
       --surface-2: #242220;
       --panel-bg: #171615;
       --overlay-bg: #201f1d;
       --border: #302e2c;
       --btn-bg: #2a2825;
       --btn-border: #4a453f;
       --btn-hover: #363330;
       --btn-active: #403c38;
       --chip-bg: #2a2825;
       --meter-bg: #2e2c29;
       --knob: #ffffff;
       --code-bg: #161514;
       --code-inline-bg: #232221;
       --logbox-bg: #0c0b0a;
       --logbox-fg: #cbc9c6;
       --error: #ff6b6b;
       --warn: #f7b015;
       --success: #9be29b;
       --tool: #f0a35a;
       --danger: #e5534b;
       --danger-fg: #ffffff;
       --scrim: rgba(0, 0, 0, 0.6);
       --shadow-1: 0 1px 2px rgba(0, 0, 0, 0.4);
       --shadow-2: 0 16px 40px rgba(0, 0, 0, 0.5);
       --tip-bg: #000000;
       --tip-fg: #ededed;
       --tip-border: #444444;
       --hl-kw: #f0a35a;
       --hl-str: #9be29b;
       --hl-num: #f7b015;
       --hl-com: #7a7a7a;
       --chart-1: #f3921a;
       --chart-2: #6ea8fe;
       --chart-3: #9be29b;
       --chart-4: #c792ea;
     }
   }
   /* Appearance preferences. They redefine size tokens on <body>, so every
      control that reads the tokens rescales together. */
   body.density-compact {
     --ctl-h: 30px; --ctl-h-sm: 24px;
     --sp-1: 3px; --sp-2: 6px; --sp-3: 9px; --sp-4: 12px; --sp-5: 18px; --sp-6: 24px;
     --header-h: 46px;
   }
   body.text-small { --fs-xs: 10px; --fs-sm: 11px; --fs-md: 13px; --fs-lg: 15px; --fs-xl: 18px; }
   body.text-large { --fs-xs: 12px; --fs-sm: 13px; --fs-md: 16px; --fs-lg: 18px; --fs-xl: 23px; }

   /* ================================================================== base */
   *, *::before, *::after { box-sizing: border-box; }
   html, body { height: 100%; }
   /* The app is a fixed frame with its own scroll regions, so the page itself
      never scrolls; that is also what keeps a stray wide child from adding a
      horizontal scrollbar to the whole window at 360px. */
   body {
     margin: 0;
     height: 100vh;
     height: 100dvh;
     display: flex;
     flex-direction: column;
     overflow: hidden;
     background: var(--bg);
     color: var(--fg);
     font-family: var(--font-sans);
     font-size: var(--fs-md);
     line-height: var(--lh);
     -webkit-font-smoothing: antialiased;
     -webkit-text-size-adjust: 100%;
     scrollbar-color: var(--btn-border) transparent;
   }
   * { scrollbar-width: thin; scrollbar-color: var(--btn-border) transparent; }
   ::selection { background: color-mix(in srgb, var(--accent) 30%, transparent); }
   h1, h2, h3, h4, h5 { margin: 0; line-height: 1.3; font-weight: 700; }
   h1 { font-size: var(--fs-xl); }
   h2 { font-size: var(--fs-lg); }
   h3, h4, h5 { font-size: var(--fs-md); }
   p { margin: 0; }
   a { color: var(--link); text-underline-offset: 2px; }
   a:hover { text-decoration-thickness: 2px; }
   code, kbd, pre, samp { font-family: var(--font-mono); }
   img, svg, video { max-width: 100%; }
   hr { border: 0; border-top: 1px solid var(--border); margin: var(--sp-3) 0; }
   :where(a, summary, [tabindex]):focus-visible {
     outline: 2px solid var(--accent);
     outline-offset: 2px;
     border-radius: var(--r-sm);
   }
   .hl-kw { color: var(--hl-kw); }
   .hl-str { color: var(--hl-str); }
   .hl-num { color: var(--hl-num); }
   .hl-com { color: var(--hl-com); font-style: italic; }

   /* ============================================================ typography */
   .h1 { font-size: var(--fs-xl); font-weight: 700; line-height: 1.3; margin: 0; }
   .h2 { font-size: var(--fs-lg); font-weight: 700; line-height: 1.3; margin: 0; }
   .h3 { font-size: var(--fs-md); font-weight: 700; line-height: 1.3; margin: 0; }
   .muted { color: var(--muted); }
   .hint { font-size: var(--fs-xs); color: var(--muted); line-height: 1.45; margin: 0; }
   .hint.warn { color: var(--warn); }
   .hint.error { color: var(--error); }
   .hint.ok { color: var(--success); }
   .mono { font-family: var(--font-mono); font-variant-numeric: tabular-nums; }
   .fs-xs { font-size: var(--fs-xs); }
   .fs-sm { font-size: var(--fs-sm); }
   .strong { font-weight: 600; }
   .nowrap { white-space: nowrap; }
   .ellipsis { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; min-width: 0; }
   .text-center { text-align: center; }
   .text-right { text-align: right; }
   .ok-text { color: var(--success); }
   .warn-text { color: var(--warn); }
   .error-text { color: var(--error); }

   /* ================================================================ layout */
   .row { display: flex; align-items: center; gap: var(--sp-2); flex-wrap: wrap; min-width: 0; }
   /* A row of labelled fields plus buttons lines the buttons up with the
      inputs, not with the middle of label-plus-input. */
   .row:has(> .field) { align-items: flex-end; }
   .row.nowrap { flex-wrap: nowrap; }
   .stack { display: flex; flex-direction: column; gap: var(--sp-3); min-width: 0; }
   .stack.tight { gap: var(--sp-2); }
   .grow { flex: 1 1 auto; min-width: 0; }
   .spacer { flex: 1 1 auto; }
   .shrink-0 { flex: 0 0 auto; }
   .wrap { flex-wrap: wrap; }
   .between { justify-content: space-between; }
   .center { justify-content: center; }
   .items-start { align-items: flex-start; }
   .items-end { align-items: flex-end; }
   .w-full { width: 100%; }
   .mt-0 { margin-top: 0; }
   .mt-1 { margin-top: var(--sp-1); }
   .mt-2 { margin-top: var(--sp-2); }
   .mt-3 { margin-top: var(--sp-3); }
   .mt-4 { margin-top: var(--sp-4); }
   .mb-2 { margin-bottom: var(--sp-2); }
   .no-max { max-height: none; }
   .scroll-y { max-height: 160px; overflow-y: auto; }
   .sr-only {
     position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px;
     overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; border: 0;
   }
   .empty {
     color: var(--muted);
     text-align: center;
     font-size: var(--fs-md);
     padding: var(--sp-6) var(--sp-4);
   }

   /* =============================================================== buttons */
   /* One button. Bare <button> elements get the same base as .btn, so markup
      and JS-built controls cannot drift apart. */
   :where(button, .btn) {
     display: inline-flex;
     align-items: center;
     justify-content: center;
     gap: var(--sp-1);
     min-height: var(--ctl-h);
     padding: 0 var(--sp-3);
     margin: 0;
     border: 1px solid var(--btn-border);
     border-radius: var(--r-md);
     background: var(--btn-bg);
     color: var(--fg);
     font-family: inherit;
     font-size: var(--fs-md);
     font-weight: 500;
     line-height: 1.2;
     white-space: nowrap;
     text-decoration: none;
     cursor: pointer;
     user-select: none;
     transition: background-color .12s ease, border-color .12s ease, color .12s ease, box-shadow .12s ease;
   }
   :where(button, .btn):where(:hover:not(:disabled)) { background: var(--btn-hover); }
   :where(button, .btn):where(:active:not(:disabled)) { background: var(--btn-active); }
   :where(button, .btn):focus-visible { outline: none; border-color: var(--accent); box-shadow: var(--focus-ring); }
   .btn-primary, button.primary {
     background: var(--accent);
     border-color: var(--accent);
     color: var(--accent-fg);
     font-weight: 600;
   }
   .btn-primary:hover:not(:disabled), button.primary:hover:not(:disabled) { background: var(--accent-hover); border-color: var(--accent-hover); }
   .btn-primary:active:not(:disabled), button.primary:active:not(:disabled) { background: var(--accent-hover); filter: brightness(0.95); }
   .btn-ghost { background: transparent; border-color: transparent; }
   .btn-ghost:hover:not(:disabled) { background: var(--btn-hover); }
   .btn-ghost:active:not(:disabled) { background: var(--btn-active); }
   .btn-danger { color: var(--danger); border-color: color-mix(in srgb, var(--danger) 45%, var(--btn-border)); }
   .btn-danger:hover:not(:disabled) { background: var(--danger); border-color: var(--danger); color: var(--danger-fg); }
   .btn-danger:active:not(:disabled) { background: var(--danger); filter: brightness(0.92); color: var(--danger-fg); }
   .btn-ghost.btn-danger { border-color: transparent; }
   .btn-danger:focus-visible { box-shadow: 0 0 0 3px color-mix(in srgb, var(--danger) 25%, transparent); border-color: var(--danger); }
   .btn-sm { min-height: var(--ctl-h-sm); padding: 0 var(--sp-2); font-size: var(--fs-sm); border-radius: var(--r-sm); }
   .btn-icon { width: var(--ctl-h); min-width: var(--ctl-h); padding: 0; }
   .btn-icon.btn-sm { width: var(--ctl-h-sm); min-width: var(--ctl-h-sm); }
   .icon { flex: 0 0 auto; display: block; }
   /* Pressed / selected (toggle buttons, a rated feedback button). */
   button.active, button[aria-pressed="true"], .btn.active, .btn[aria-pressed="true"] {
     background: var(--accent-soft);
     border-color: var(--accent);
     color: var(--fg);
   }
   button:disabled, .btn:disabled, .btn[aria-disabled="true"] {
     opacity: 0.5;
     cursor: not-allowed;
     box-shadow: none;
   }
   /* A rated feedback button is disabled AND pressed; it must stay legible. */
   button[aria-pressed="true"]:disabled, button.active:disabled { opacity: 0.85; }

   /* ================================================================ inputs */
   :where(.input, select, textarea,
     input:not([type=checkbox]):not([type=radio]):not([type=range]):not([type=file]):not([type=color]):not([type=hidden]):not([type=button]):not([type=submit])) {
     min-height: var(--ctl-h);
     max-width: 100%;
     min-width: 0;
     padding: 0 var(--sp-3);
     margin: 0;
     border: 1px solid var(--border);
     border-radius: var(--r-md);
     background: var(--surface);
     color: var(--fg);
     font-family: inherit;
     font-size: var(--fs-md);
     line-height: 1.3;
     outline: none;
     transition: border-color .12s ease, box-shadow .12s ease;
   }
   :where(textarea, .textarea) {
     min-height: 80px;
     padding: var(--sp-2) var(--sp-3);
     line-height: var(--lh);
     resize: vertical;
   }
   :where(.input, select, textarea, input:not([type=checkbox]):not([type=radio]):not([type=range]):not([type=file]):not([type=color]):not([type=hidden]):not([type=button]):not([type=submit])):where(:hover:not(:disabled):not(:focus)) { border-color: var(--btn-border); }
   :where(.input, select, textarea, input:not([type=checkbox]):not([type=radio]):not([type=range]):not([type=file]):not([type=color]):not([type=hidden]):not([type=button]):not([type=submit])):focus { border-color: var(--accent); box-shadow: var(--focus-ring); }
   :where(.input, select, textarea, input:not([type=checkbox]):not([type=radio]):not([type=range]):not([type=file]):not([type=color]):not([type=hidden]):not([type=button]):not([type=submit])):disabled { opacity: 0.6; cursor: not-allowed; background-color: var(--surface-2); }
   :where(.input, textarea, input)[readonly] { background-color: var(--surface-2); }
   .input.invalid, .input[aria-invalid="true"] { border-color: var(--error); }
   ::placeholder { color: var(--muted); opacity: 1; }
   input[type=search]::-webkit-search-decoration { -webkit-appearance: none; }
   /* Native select arrows differ per OS and ignore the theme; a two-gradient
      chevron uses the muted token instead of a hard-coded image colour. */
   :where(select:not([multiple]), .select) {
     appearance: none;
     -webkit-appearance: none;
     padding-right: 30px;
     background-image:
       linear-gradient(45deg, transparent 50%, var(--muted) 50%),
       linear-gradient(135deg, var(--muted) 50%, transparent 50%);
     background-position: calc(100% - 16px) 50%, calc(100% - 11px) 50%;
     background-size: 5px 5px, 5px 5px;
     background-repeat: no-repeat;
     cursor: pointer;
   }
   :where(select[multiple]) { padding: var(--sp-1); }
   :where(input[type=checkbox], input[type=radio]) {
     accent-color: var(--accent);
     width: 16px;
     height: 16px;
     margin: 0;
     flex: 0 0 auto;
     cursor: pointer;
   }
   :where(input[type=checkbox], input[type=radio]):focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
   .unit { color: var(--muted); font-size: var(--fs-sm); white-space: nowrap; }

   /* ================================================================ switch */
   /* The only boolean control. The real checkbox stays in the box (invisible)
      so it keeps keyboard focus, form semantics and screen reader state. */
   .switch { position: relative; display: inline-flex; flex: 0 0 auto; width: 36px; height: 20px; vertical-align: middle; }
   .switch input { position: absolute; inset: 0; width: 100%; height: 100%; margin: 0; opacity: 0; cursor: pointer; z-index: 1; }
   .switch .slider {
     position: absolute; inset: 0; border-radius: var(--r-pill);
     background: var(--btn-border); transition: background-color .15s ease, box-shadow .15s ease;
   }
   .switch .slider::before {
     content: ""; position: absolute; left: 3px; top: 3px; width: 14px; height: 14px;
     border-radius: 50%; background: var(--knob); box-shadow: var(--shadow-1);
     transition: transform .15s ease;
   }
   .switch:hover input:not(:disabled) + .slider { background: color-mix(in srgb, var(--btn-border) 80%, var(--fg)); }
   .switch input:checked + .slider { background: var(--accent); }
   .switch:hover input:checked:not(:disabled) + .slider { background: var(--accent-hover); }
   .switch input:checked + .slider::before { transform: translateX(16px); }
   .switch input:focus-visible + .slider { box-shadow: var(--focus-ring); }
   .switch input:disabled { cursor: not-allowed; }
   .switch input:disabled + .slider { opacity: 0.5; }

   /* ========================================================= check (rows) */
   /* Styled checkbox row for multi-select lists (capabilities, document
      scope, agent picker). .cap-item is the same component under its older
      name. */
   .check, .cap-item {
     display: flex; align-items: flex-start; gap: var(--sp-2);
     padding: var(--sp-2) var(--sp-3);
     border: 1px solid var(--border); border-radius: var(--r-md);
     background: var(--surface);
     font-size: var(--fs-md); line-height: 1.35;
     cursor: pointer; user-select: none; min-width: 0;
     transition: border-color .12s ease, background-color .12s ease;
   }
   .check input, .cap-item input { margin-top: 1px; }
   .check:hover, .cap-item:hover { border-color: var(--btn-border); }
   .check:has(input:checked), .cap-item:has(input:checked) { border-color: var(--accent); background: var(--accent-soft); }
   .check:has(input:focus-visible), .cap-item:has(input:focus-visible) { box-shadow: var(--focus-ring); }
   .check:has(input:disabled), .cap-item:has(input:disabled) { opacity: 0.55; cursor: not-allowed; }
   .cap-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: var(--sp-2); }
   .cap-text { display: flex; flex-direction: column; min-width: 0; }
   .cap-label { font-size: var(--fs-md); font-weight: 600; }
   .cap-desc { font-size: var(--fs-xs); color: var(--muted); margin-top: 2px; }

   /* ================================================================= range */
   /* One slider everywhere. JS keeps --fill (0-100%) on the input so the
      track left of the thumb shows the accent in WebKit too (Firefox has a
      native progress pseudo-element). */
   .range { display: flex; align-items: center; gap: var(--sp-3); min-width: 0; width: 100%; }
   .range input[type=range] { flex: 1 1 auto; }
   .range-value {
     min-width: 5ch; text-align: right; font-family: var(--font-mono);
     font-size: var(--fs-sm); color: var(--fg); font-variant-numeric: tabular-nums;
   }
   input[type=range] {
     -webkit-appearance: none; appearance: none;
     width: 100%; min-width: 0; height: 20px; margin: 0;
     background: transparent; cursor: pointer;
   }
   input[type=range]:focus { outline: none; }
   input[type=range]::-webkit-slider-runnable-track {
     height: 4px; border-radius: var(--r-pill);
     background: linear-gradient(to right, var(--accent) 0, var(--accent) var(--fill, 0%), var(--meter-bg) var(--fill, 0%), var(--meter-bg) 100%);
   }
   input[type=range]::-webkit-slider-thumb {
     -webkit-appearance: none; appearance: none;
     width: 16px; height: 16px; margin-top: -6px; border-radius: 50%;
     background: var(--knob); border: 2px solid var(--accent); box-shadow: var(--shadow-1);
     transition: box-shadow .12s ease, transform .12s ease;
   }
   input[type=range]:hover::-webkit-slider-thumb { transform: scale(1.1); }
   input[type=range]:focus-visible::-webkit-slider-thumb { box-shadow: var(--focus-ring); }
   input[type=range]::-moz-range-track { height: 4px; border-radius: var(--r-pill); background: var(--meter-bg); }
   input[type=range]::-moz-range-progress { height: 4px; border-radius: var(--r-pill); background: var(--accent); }
   input[type=range]::-moz-range-thumb {
     width: 12px; height: 12px; border-radius: 50%;
     background: var(--knob); border: 2px solid var(--accent); box-shadow: var(--shadow-1);
   }
   input[type=range]:focus-visible::-moz-range-thumb { box-shadow: var(--focus-ring); }
   input[type=range]:disabled { opacity: 0.5; cursor: not-allowed; }

   /* ===================================================== segmented control */
   .seg {
     display: inline-flex; align-items: stretch; gap: 2px; padding: 2px;
     max-width: 100%; min-width: 0;
     background: var(--surface-2); border: 1px solid var(--border); border-radius: var(--r-md);
   }
   .seg > button {
     flex: 0 0 auto;
     min-height: calc(var(--ctl-h) - 6px);
     padding: 0 var(--sp-3);
     border: 1px solid transparent;
     border-radius: calc(var(--r-md) - 3px);
     background: transparent;
     color: var(--muted);
     font-size: var(--fs-sm);
     font-weight: 600;
   }
   .seg > button:hover:not(:disabled) { background: var(--btn-hover); color: var(--fg); }
   .seg > button.active, .seg > button[aria-pressed="true"], .seg > button[aria-selected="true"], .seg > button[aria-checked="true"] {
     background: var(--btn-bg); border-color: var(--border); color: var(--fg); box-shadow: var(--shadow-1);
   }
   /* Sub-tabs: the same control, full width and scrolling sideways instead
      of wrapping when there are more tabs than room. */
   .seg-tabs { display: flex; overflow-x: auto; scrollbar-width: none; align-self: flex-start; }
   .seg-tabs::-webkit-scrollbar { display: none; }

   /* ================================================================== menu */
   .menu {
     position: absolute; z-index: var(--z-pop);
     display: flex; flex-direction: column; gap: 2px;
     min-width: 210px; max-width: min(320px, calc(100vw - 2 * var(--sp-4)));
     padding: var(--sp-1);
     background: var(--overlay-bg); color: var(--fg);
     border: 1px solid var(--border); border-radius: var(--r-md);
     box-shadow: var(--shadow-2);
   }
   .menu > button, .menu [role=menuitem] {
     width: 100%; justify-content: flex-start; text-align: left;
     border-color: transparent; background: transparent;
     border-radius: var(--r-sm); font-weight: 500;
   }
   .menu > button:hover:not(:disabled), .menu [role=menuitem]:hover:not(:disabled) { background: var(--btn-hover); }
   .menu-sep { height: 1px; background: var(--border); margin: var(--sp-1) 0; }

   /* ================================================================ toasts */
   .toast-host {
     position: fixed; z-index: calc(var(--z-modal) + 5);
     top: calc(var(--header-h) + var(--sp-2)); right: var(--sp-4);
     display: flex; flex-direction: column; align-items: flex-end; gap: var(--sp-2);
     width: min(380px, calc(100vw - 2 * var(--sp-4)));
     pointer-events: none;
   }
   .toast {
     pointer-events: auto; max-width: 100%;
     display: flex; align-items: flex-start; gap: var(--sp-2);
     padding: var(--sp-2) var(--sp-2) var(--sp-2) var(--sp-3);
     background: var(--overlay-bg); color: var(--fg);
     border: 1px solid var(--border); border-left: 3px solid var(--accent);
     border-radius: var(--r-md); box-shadow: var(--shadow-2);
     font-size: var(--fs-sm); line-height: 1.45; overflow-wrap: anywhere; white-space: pre-wrap;
     animation: toast-in .18s ease-out;
     transition: opacity .2s ease, transform .2s ease;
   }
   .toast-text { flex: 1 1 auto; min-width: 0; padding: 3px 0; }
   .toast-close { flex: 0 0 auto; white-space: normal; color: var(--muted); }
   .toast.ok { border-left-color: var(--success); }
   .toast.warn { border-left-color: var(--warn); }
   .toast.error { border-left-color: var(--error); }
   .toast.info { border-left-color: var(--accent); }
   .toast.out, .toast.leaving { opacity: 0; transform: translateY(-4px); }
   @keyframes toast-in { from { opacity: 0; transform: translateY(-6px); } }

   /* ========================================================= card / pills */
   .card {
     display: flex; flex-direction: column; gap: var(--sp-2);
     min-width: 0; padding: var(--sp-3) var(--sp-4);
     background: var(--surface); border: 1px solid var(--border); border-radius: var(--r-lg);
   }
   .card-header { display: flex; align-items: center; gap: var(--sp-2); min-width: 0; }
   .card-header > :first-child { min-width: 0; }
   .card-actions { display: flex; flex-wrap: wrap; align-items: center; gap: var(--sp-1); }
   .card-header { flex-wrap: wrap; }
   .card-header > .card-actions { margin-left: auto; }
   .card.selected { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); }
   .card.failed { border-color: color-mix(in srgb, var(--error) 60%, var(--border)); }
   .card h4 { margin: 0; font-size: var(--fs-md); }
   .card .goal { font-size: var(--fs-sm); color: var(--muted); line-height: 1.45; max-height: 3.2em; overflow: hidden; }
   .card .meta { font-size: var(--fs-xs); color: var(--muted); font-family: var(--font-mono); }
   .pill {
     display: inline-flex; align-items: center; gap: var(--sp-1);
     padding: 1px var(--sp-2); border-radius: var(--r-pill);
     border: 1px solid var(--btn-border); color: var(--muted);
     font-family: var(--font-mono); font-size: var(--fs-xs); line-height: 1.5;
     white-space: nowrap; max-width: 100%; overflow: hidden; text-overflow: ellipsis;
   }
   .pill.ok { border-color: var(--success); color: var(--success); }
   .pill.warn, .pill.cancelled { border-color: var(--warn); color: var(--warn); }
   .pill.error, .pill.interrupted, .pill.failed { border-color: var(--error); color: var(--error); }
   .pill.running { border-color: var(--accent); color: var(--accent); }
   .pill.muted { border-color: var(--btn-border); color: var(--muted); }
   .badge {
     display: inline-flex; align-items: center;
     padding: 0 6px; border-radius: var(--r-sm);
     background: var(--chip-bg); border: 1px solid var(--border); color: var(--muted);
     font-size: var(--fs-xs); font-weight: 600; line-height: 1.6; letter-spacing: .02em;
     white-space: nowrap; vertical-align: middle;
   }
   .badge.warn, .field-label .badge {
     color: var(--warn);
     border-color: color-mix(in srgb, var(--warn) 40%, transparent);
     background: color-mix(in srgb, var(--warn) 10%, transparent);
   }
   .badge.ok { color: var(--success); border-color: color-mix(in srgb, var(--success) 40%, transparent); }
   .badge.error { color: var(--error); border-color: color-mix(in srgb, var(--error) 40%, transparent); }

   /* ======================================================= table / data */
   .table, table.models { width: 100%; border-collapse: collapse; font-size: var(--fs-md); }
   .table th, table.models th {
     text-align: left; font-size: var(--fs-xs); font-weight: 700; color: var(--muted);
     text-transform: uppercase; letter-spacing: .04em;
     padding: var(--sp-2); border-bottom: 1px solid var(--border);
   }
   .table td, table.models td { padding: var(--sp-2); border-bottom: 1px solid var(--border); vertical-align: middle; overflow-wrap: anywhere; }
   .table tr:last-child td, table.models tr:last-child td { border-bottom: 0; }
   .table tbody tr:hover td, table.models tr:hover td { background: var(--surface-2); }
   .table td.cell-status, .table td.cell-action { width: 1%; white-space: nowrap; }
   .table td.cell-action { text-align: right; }
   .table td.id, table.models td.id, .table .mono { font-family: var(--font-mono); font-size: var(--fs-sm); word-break: break-all; }
   .readonly {
     display: flex; align-items: center; min-height: var(--ctl-h); width: 100%; min-width: 0;
     padding: var(--sp-1) var(--sp-3);
     background: var(--surface-2); color: var(--fg);
     border: 1px solid var(--border); border-radius: var(--r-md);
     font-size: var(--fs-sm); overflow-wrap: anywhere;
   }
   .kv {
     display: grid; grid-template-columns: minmax(0, max-content) minmax(0, 1fr);
     gap: var(--sp-1) var(--sp-4); margin: 0; font-size: var(--fs-sm);
   }
   .kv > :nth-child(odd), .kv > dt { color: var(--muted); }
   .kv > :nth-child(even), .kv > dd { margin: 0; overflow-wrap: anywhere; }
   .meter { height: 6px; background: var(--meter-bg); border-radius: var(--r-pill); overflow: hidden; }
   .meter > div { height: 100%; width: 0%; background: var(--accent); border-radius: inherit; transition: width .3s ease; }
   .logbox {
     max-height: 260px; overflow: auto; margin: 0; padding: var(--sp-2) var(--sp-3);
     background: var(--logbox-bg); color: var(--logbox-fg);
     border: 1px solid var(--border); border-radius: var(--r-md);
     font-family: var(--font-mono); font-size: var(--fs-xs); line-height: 1.5; white-space: pre-wrap; overflow-wrap: anywhere;
   }
   .tools-list { font-size: var(--fs-xs); color: var(--muted); font-family: var(--font-mono); line-height: 1.6; }

   /* ============================================================== tooltips */
   /* Any element with data-tip explains itself on hover (mouse only: on a
      touch screen a tap would leave the bubble stuck over the content) and
      on keyboard focus. Only ::after is used, so an element's own ::before
      (a caret, a status dot) is never hijacked. */
   [data-tip] { position: relative; }
   [data-tip]:focus-visible::after {
     content: attr(data-tip);
     position: absolute; left: 50%; bottom: calc(100% + 8px); transform: translateX(-50%);
     z-index: var(--z-tip); width: max-content; max-width: 260px;
     padding: 6px 9px; border-radius: var(--r-sm);
     background: var(--tip-bg); color: var(--tip-fg); border: 1px solid var(--tip-border);
     box-shadow: var(--shadow-2);
     font-family: var(--font-sans); font-size: var(--fs-sm); font-weight: 400; line-height: 1.4;
     letter-spacing: normal; text-transform: none; text-align: left; white-space: normal;
     pointer-events: none;
   }
   @media (hover: hover) {
     [data-tip]:hover::after {
       content: attr(data-tip);
       position: absolute; left: 50%; bottom: calc(100% + 8px); transform: translateX(-50%);
       z-index: var(--z-tip); width: max-content; max-width: 260px;
       padding: 6px 9px; border-radius: var(--r-sm);
       background: var(--tip-bg); color: var(--tip-fg); border: 1px solid var(--tip-border);
       box-shadow: var(--shadow-2);
       font-family: var(--font-sans); font-size: var(--fs-sm); font-weight: 400; line-height: 1.4;
       letter-spacing: normal; text-transform: none; text-align: left; white-space: normal;
       pointer-events: none;
     }
   }
   [data-tip-below]:hover::after, [data-tip-below]:focus-visible::after { bottom: auto; top: calc(100% + 8px); }
   /* Controls at the right edge anchor their bubble to the right so it does
      not run off the window. */
   body > header .actions [data-tip]:hover::after, body > header .actions [data-tip]:focus-visible::after,
   #chatMenuBtn:hover::after, #chatMenuBtn:focus-visible::after,
   .composer-toolbar #sendBtn:hover::after, .composer-toolbar #sendBtn:focus-visible::after {
     left: auto; right: 0; transform: none;
   }

   /* ================================================================ header */
   body > header {
     flex: 0 0 auto;
     position: relative; z-index: 20;
     display: flex; align-items: center; gap: var(--sp-3);
     min-height: var(--header-h); min-width: 0;
     padding: var(--sp-2) var(--sp-4);
     background: var(--surface);
     border-bottom: 1px solid var(--border);
   }
   .brand { display: flex; align-items: center; gap: var(--sp-2); flex: 0 0 auto; min-width: 0; }
   .brand-slot { display: inline-flex; align-items: center; }
   .brand-logo { height: 24px; width: auto; aspect-ratio: 160.03 / 183.2; display: block; }
   .brand-mark { font-size: 18px; line-height: 1; color: var(--accent); }
   .brand strong { font-weight: 800; font-size: var(--fs-md); letter-spacing: .09em; text-transform: uppercase; white-space: nowrap; }
   .brand .build { color: var(--muted); font-size: var(--fs-xs); white-space: nowrap; }
   body > header .seg { flex: 0 1 auto; }
   /* One line, ellipsised. The full detail lives in the pill's tooltip, so
      the header stays a single row; min-width:0 is what lets a flex item
      shrink below its content width. */
   #status { flex: 0 1 auto; min-width: 0; max-width: 360px; }
   header .actions { display: flex; align-items: center; gap: var(--sp-1); flex: 0 0 auto; min-width: 0; }
   .acct { position: relative; }
   .acct-btn { gap: 6px; max-width: 220px; }
   .acct-btn .caret { color: var(--muted); font-size: 10px; }
   .user-chip { font-size: var(--fs-sm); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; min-width: 0; }
   .acct-pop { right: 0; top: calc(100% + 6px); }
   .acct-info {
     padding: var(--sp-2) var(--sp-3); margin-bottom: var(--sp-1);
     border-bottom: 1px solid var(--border);
     font-size: var(--fs-sm); color: var(--muted); line-height: 1.5; overflow-wrap: anywhere;
   }
   .acct-info strong { color: var(--fg); }

   /* ================================================================== main */
   body > #main { flex: 1 1 auto; display: flex; min-height: 0; min-width: 0; overflow: hidden; position: relative; }
   .view { flex: 1 1 auto; display: flex; min-width: 0; min-height: 0; overflow: hidden; }
   .view.hidden { display: none; }
   .panel { flex: 1 1 auto; min-width: 0; overflow-y: auto; padding: var(--sp-4); }

   /* ================================================================== chat */
   #chatView { position: relative; flex-direction: column; }
   /* The column is centred with padding rather than a centred box, so the
      scrollbar stays at the window edge where the hand expects it. */
   #chat {
     flex: 1 1 auto; min-height: 0; overflow-y: auto; overflow-x: hidden;
     display: flex; flex-direction: column; gap: var(--sp-3);
     padding: var(--sp-5) max(var(--sp-4), calc((100% - 820px) / 2)) var(--sp-4);
   }
   #chatMenuBtn {
     position: absolute; top: var(--sp-2); right: var(--sp-3); z-index: 6;
     background: var(--bg); border-color: var(--border);
   }
   #chatMenuBtn:hover:not(:disabled) { background: var(--btn-hover); }
   #chatMenu { top: calc(var(--sp-2) + var(--ctl-h) + 4px); right: var(--sp-3); }
   .chat-empty { margin: auto; max-width: 440px; }
   .msg {
     max-width: 85%; min-width: 0;
     padding: var(--sp-2) var(--sp-3);
     border-radius: var(--r-lg);
     font-size: var(--fs-md); line-height: var(--lh);
     white-space: pre-wrap; overflow-wrap: anywhere;
   }
   .msg.user { align-self: flex-end; background: var(--accent); color: var(--accent-fg); border-bottom-right-radius: var(--r-sm); }
   .msg.assistant { align-self: flex-start; background: var(--surface); border: 1px solid var(--border); border-bottom-left-radius: var(--r-sm); }
   .msg.assistant.pending, .assistant.pending { opacity: 0.75; }
   .msg-body { min-width: 0; }
   .msg-body a, .gtext a { color: var(--link); }
   .system-msg { align-self: center; color: var(--muted); font-size: var(--fs-sm); text-align: center; margin: var(--sp-1) 0; }
   /* Per-message actions: quiet until the message is hovered or focused
      (always shown on touch screens and on the latest message). */
   .msg-actions { display: flex; flex-wrap: wrap; gap: 2px; margin-top: var(--sp-1); white-space: normal; }
   .msg-actions .btn, .msg-actions button {
     min-height: 24px; padding: 0 var(--sp-2); border-radius: var(--r-sm);
     font-size: var(--fs-xs); color: var(--muted); background: transparent; border-color: transparent;
   }
   .msg-actions .btn:hover:not(:disabled), .msg-actions button:hover:not(:disabled) { color: var(--fg); background: var(--btn-hover); }
   .msg.user .msg-actions .btn, .msg.user .msg-actions button { color: inherit; opacity: 0.75; }
   .msg.user .msg-actions .btn:hover:not(:disabled), .msg.user .msg-actions button:hover:not(:disabled) { opacity: 1; background: color-mix(in srgb, var(--accent-fg) 10%, transparent); }
   @media (hover: hover) {
     #chat .msg > .msg-actions { opacity: 0; transition: opacity .12s ease; }
     #chat .msg:hover > .msg-actions, #chat .msg:focus-within > .msg-actions, #chat .msg:last-child > .msg-actions { opacity: 1; }
   }
   .tool-card {
     align-self: flex-start; max-width: 85%; min-width: 0;
     padding: var(--sp-2) var(--sp-3);
     background: var(--surface-2); border: 1px solid var(--tool); border-radius: var(--r-md);
     font-family: var(--font-mono); font-size: var(--fs-sm);
   }
   .tool-card .name { color: var(--tool); font-weight: 600; }
   .tool-card .args { color: var(--muted); margin: var(--sp-1) 0; white-space: pre-wrap; overflow-wrap: anywhere; }
   .tool-card pre { margin: 6px 0 0; white-space: pre-wrap; max-height: 220px; overflow: auto; color: var(--fg); }
   .tool-card.failed { border-color: var(--error); }

   /* ================================================== agent trace (glass box) */
   .turn { align-self: stretch; display: flex; flex-direction: column; gap: var(--sp-2); min-width: 0; }
   .trace {
     align-self: flex-start; max-width: 92%; min-width: 0;
     margin-left: 6px; padding-left: 14px;
     border-left: 2px solid var(--border);
     display: flex; flex-direction: column; gap: 10px;
   }
   .trace:empty { display: none; }
   .gnode { position: relative; font-size: var(--fs-md); min-width: 0; }
   .gnode::before {
     content: ""; position: absolute; left: -19px; top: 5px;
     width: 8px; height: 8px; border-radius: 50%;
     background: var(--surface); border: 1.5px solid var(--border);
   }
   .gnode.router::before { border-color: var(--accent); }
   .gnode.ok::before { border-color: var(--success); background: var(--success); }
   .gnode.failed::before { border-color: var(--error); background: var(--error); }
   .gnode.notice::before { border-color: var(--warn); background: var(--warn); }
   .gnode .ghead { display: flex; align-items: center; gap: var(--sp-2); min-width: 0; color: var(--muted); font-family: var(--font-mono); font-size: var(--fs-sm); }
   .gnode.tool .ghead { cursor: pointer; user-select: none; }
   .gnode .gtool { color: var(--tool); font-weight: 600; }
   .gnode .gpill {
     font-size: var(--fs-xs); padding: 1px 7px; border-radius: var(--r-pill);
     background: color-mix(in srgb, var(--success) 16%, transparent); color: var(--success);
   }
   .gnode.failed .gpill { background: color-mix(in srgb, var(--error) 16%, transparent); color: var(--error); }
   .gnode.tool.running .gpill { background: var(--accent-soft); color: var(--accent); }
   .gnode .gcaret { margin-left: auto; color: var(--muted); transition: transform .15s ease; }
   .gnode.open .gcaret { transform: rotate(90deg); }
   .gnode .gargs { color: var(--muted); font-family: var(--font-mono); font-size: var(--fs-sm); margin-top: 3px; overflow-wrap: anywhere; }
   .gbody {
     margin: 6px 0 0; padding: var(--sp-2) 10px; max-height: 240px; overflow: auto;
     background: var(--logbox-bg); color: var(--logbox-fg); border-radius: var(--r-sm);
     font-family: var(--font-mono); font-size: var(--fs-sm); white-space: pre-wrap; overflow-wrap: anywhere;
   }
   .gnode.tool:not(.open) .gbody { display: none; }
   .gnode.notice .gmsg { color: var(--warn); }
   .gnode.notice.info::before { border-color: var(--tool); background: var(--tool); }
   .gnode.notice.info .gmsg { color: var(--tool); }
   .gnode.reason::before { border-color: var(--accent); }
   .gnode.reason .gbody { color: var(--muted); max-height: 160px; }
   .gnode.reason.done::before { border-color: var(--success); background: var(--success); }
   /* A finished step clamps to about two lines and reopens on a click. Six
      open 160px panes pushed the answer a screen and a half down the page. */
   .gnode.reason.done .ghead { cursor: pointer; user-select: none; }
   .gnode.reason.done .gbody { max-height: 3.2em; overflow: hidden; }
   .gnode.reason.done.open .gbody { max-height: 160px; overflow: auto; }
   .gnode.thinking::before { border-color: var(--muted); }
   .gnode.thinking .ghead { cursor: pointer; user-select: none; }
   .gnode.thinking .gtool { color: var(--muted); }
   .gnode.thinking .gbody { color: var(--muted); font-style: italic; max-height: 200px; }
   .gnode.thinking:not(.open) .gbody { display: none; }
   .gnode.diff .gtool { color: var(--accent); }
   .gnode.diff:not(.open) .gbody { display: none; }
   .gdiff { max-height: 340px; overflow: auto; font-size: var(--fs-xs); line-height: 1.4; }
   .gdetail { margin: 1px 0 1px 22px; font-family: var(--font-mono); font-size: var(--fs-xs); color: var(--muted); white-space: pre-wrap; }
   .gdetail::before { content: "\2699 "; opacity: 0.6; }
   /* The answer is the turn's result, not a step: no timeline dot. */
   .gnode.answer::before { content: none; }
   .gnode.answer .gtext {
     padding: var(--sp-2) var(--sp-3);
     background: var(--surface); color: var(--fg);
     border: 1px solid var(--border); border-radius: var(--r-lg); border-bottom-left-radius: var(--r-sm);
     font-size: var(--fs-md); line-height: var(--lh); white-space: pre-wrap; overflow-wrap: anywhere;
   }
   .gnode.answer.pending .gtext { opacity: 0.75; }
   .gnode.answer.failed .gtext { border-color: color-mix(in srgb, var(--error) 50%, var(--border)); }
   .gmeta { color: var(--muted); font-size: var(--fs-xs); font-family: var(--font-mono); margin: 2px 0 0 6px; }
   /* One line, always the same height, never wider than the column: a long
      notice used to wrap to four lines and snap back on the next event,
      moving the page under text being read. tabular-nums keeps the clock
      from nudging the label sideways as digits change. */
   .gactivity {
     display: flex; align-items: center; gap: var(--sp-2);
     min-height: 17px; min-width: 0; margin: 2px 0 0 6px;
     font-family: var(--font-mono); font-size: var(--fs-sm); color: var(--accent);
     transition: color .45s ease;
   }
   .gactivity .glabel { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; min-width: 0; }
   .gactivity .gelapsed { color: var(--muted); font-variant-numeric: tabular-nums; }
   .gactivity.stalled { color: var(--warn); }
   .gactivity.stalled .gspin { border-top-color: var(--warn); }
   .gspin {
     flex: 0 0 auto; width: 11px; height: 11px; border-radius: 50%;
     border: 2px solid var(--border); border-top-color: var(--accent);
     animation: gspin .7s linear infinite;
   }
   @keyframes gspin { to { transform: rotate(360deg); } }
   body.hide-metrics .gmeta, body.hide-metrics .composer-meta { display: none; }

   /* ============================================================= feedback */
   .feedback { align-self: flex-start; display: flex; flex-wrap: wrap; gap: var(--sp-1); margin: 0 0 var(--sp-2) 6px; }
   .feedback .btn, .feedback button { color: var(--muted); }
   .feedback .btn:hover:not(:disabled), .feedback button:hover:not(:disabled) { color: var(--fg); }
   .feedback .btn.active, .feedback button.active, .feedback button[aria-pressed="true"] { color: var(--fg); }

   /* ============================================================= markdown */
   .md-p { margin: 0 0 10px; line-height: 1.55; white-space: pre-wrap; }
   .md-h { margin: var(--sp-3) 0 6px; font-size: var(--fs-md); font-weight: 700; }
   .md-list { margin: 0 0 10px; padding-left: 22px; line-height: 1.55; }
   .md-list li { margin: 2px 0; }
   .md-quote { margin: 0 0 10px; padding: 6px var(--sp-3); border-left: 3px solid var(--border); color: var(--muted); }
   .msg-body > :last-child, .gtext > :last-child { margin-bottom: 0; }
   .inline-code {
     padding: 1px 5px; border-radius: 4px;
     background: var(--code-inline-bg); border: 1px solid var(--border);
     font-family: var(--font-mono); font-size: 0.86em;
   }
   .codeblock { margin: 0 0 10px; overflow: hidden; background: var(--code-bg); border: 1px solid var(--border); border-radius: var(--r-md); white-space: normal; }
   .codebar {
     display: flex; justify-content: space-between; align-items: center; gap: var(--sp-2);
     padding: 3px var(--sp-1) 3px var(--sp-3);
     background: var(--chip-bg); border-bottom: 1px solid var(--border);
     font-family: var(--font-mono); font-size: var(--fs-xs); color: var(--muted);
   }
   .codebar .btn, .codebar button { min-height: 22px; padding: 0 var(--sp-2); font-size: var(--fs-xs); border-radius: var(--r-sm); color: var(--muted); }
   .codebar .btn:hover:not(:disabled), .codebar button:hover:not(:disabled) { color: var(--fg); }
   .codeblock pre {
     margin: 0; padding: 10px var(--sp-3); overflow-x: auto;
     font-family: var(--font-mono); font-size: var(--fs-sm); line-height: 1.45; white-space: pre;
   }
   body.code-wrap .codeblock pre { white-space: pre-wrap; overflow-wrap: anywhere; }
   .clip-buffer { position: fixed; top: 0; left: -9999px; width: 1px; height: 1px; opacity: 0; pointer-events: none; }

   /* ============================================== generated-document links */
   .docrow { display: flex; flex-wrap: wrap; gap: var(--sp-2); margin: 10px 0 2px; }
   .docdl {
     display: inline-flex; align-items: center; gap: 7px;
     min-height: var(--ctl-h); padding: 0 var(--sp-3);
     border: 1px solid var(--btn-border); border-radius: var(--r-md);
     background: var(--btn-bg); color: var(--link);
     font-size: var(--fs-md); font-weight: 600; text-decoration: none;
     transition: background-color .12s ease, border-color .12s ease;
   }
   .docdl:hover { background: var(--btn-hover); border-color: var(--accent); }
   .docdl:active { transform: translateY(1px); }
   .docdl:focus-visible { outline: none; box-shadow: var(--focus-ring); border-color: var(--accent); }

   /* ============================================================= composer */
   body > #composer {
     flex: 0 0 auto;
     display: flex; justify-content: center;
     padding: var(--sp-2) var(--sp-4) var(--sp-3);
     background: var(--bg);
   }
   .composer-box {
     flex: 1 1 auto; width: 100%; max-width: 820px; min-width: 0;
     display: flex; flex-direction: column;
     background: var(--surface); border: 1px solid var(--border); border-radius: var(--r-lg);
     box-shadow: var(--shadow-1);
     transition: border-color .12s ease, box-shadow .12s ease;
   }
   .composer-box:has(#input:focus) { border-color: var(--accent); box-shadow: var(--focus-ring); }
   #input {
     width: 100%; min-height: 46px; max-height: 200px;
     padding: var(--sp-3) var(--sp-3) var(--sp-1);
     border: 0; border-radius: var(--r-lg) var(--r-lg) 0 0;
     background: transparent; box-shadow: none; resize: none;
   }
   #input:focus { box-shadow: none; }
   .composer-toolbar { display: flex; align-items: center; flex-wrap: wrap; gap: var(--sp-2); min-width: 0; padding: var(--sp-1) var(--sp-2) var(--sp-2); }
   .ctx-meter { display: flex; align-items: center; gap: var(--sp-2); min-width: 0; }
   .ctx-meter .meter { width: 72px; flex: 0 0 auto; }
   #ctxHint { font-size: var(--fs-xs); white-space: nowrap; }
   /* Stop/Send stay right-aligned even when the toolbar wraps on a phone. */
   .composer-toolbar #stopBtn { margin-left: auto; }
   /* The metrics line takes the free space with a zero basis and ellipsizes:
      with an auto basis its text width made the wrapping toolbar push Send onto
      a second row once the mic and pickers joined it. Stop's auto margin keeps
      the buttons right-aligned, so the spacer is redundant here. */
   .composer-meta { flex: 1 1 0; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: var(--fs-xs); }
   .composer-toolbar > .spacer { display: none; }

   /* =============================================================== history */
   #historyView > .panel { width: 100%; max-width: 860px; margin: 0 auto; padding: var(--sp-5) var(--sp-4); }
   #historyList, #promptList, #taskCards { gap: var(--sp-2); }
   .conv-item { padding: 10px var(--sp-3); gap: 2px; }
   .conv-item:hover { border-color: var(--btn-border); }
   .conv-item .ctitle { font-size: var(--fs-md); font-weight: 600; overflow-wrap: anywhere; }
   .conv-item .cmeta { font-size: var(--fs-xs); color: var(--muted); }
   .conv-item .csnip { font-size: var(--fs-sm); color: var(--muted); margin-top: var(--sp-1); overflow-wrap: anywhere; }
   .conv-item .msg-actions { margin-top: var(--sp-1); }
   .memory-item {
     justify-content: space-between; flex-wrap: nowrap;
     padding: var(--sp-2) var(--sp-3);
     background: var(--surface); border: 1px solid var(--border); border-radius: var(--r-md);
     font-size: var(--fs-sm); overflow-wrap: anywhere;
   }
   .memory-item > :first-child { min-width: 0; }

   /* ====================================================== prompts aside */
   #promptsPanel {
     flex: 0 0 340px; width: 340px; min-width: 0;
     padding: var(--sp-4); overflow-y: auto;
     background: var(--panel-bg); border-left: 1px solid var(--border);
   }
   .prompt-item { padding: var(--sp-2) var(--sp-3); background: var(--surface); border: 1px solid var(--border); border-radius: var(--r-md); }
   .prompt-item .pname { font-size: var(--fs-md); font-weight: 600; }
   .prompt-item .pbody { font-size: var(--fs-sm); color: var(--muted); margin: var(--sp-1) 0 6px; max-height: 3.4em; overflow: hidden; overflow-wrap: anywhere; }

   /* ==================================================== palette / browser */
   #palette, #browser {
     position: fixed; inset: 0; z-index: var(--z-modal);
     display: flex; align-items: flex-start; justify-content: center;
     padding: 12vh var(--sp-4) var(--sp-4);
     background: var(--scrim);
   }
   #browser { padding-top: 10vh; }
   #paletteBox, #browserBox {
     width: min(560px, 100%); max-height: 100%;
     display: flex; flex-direction: column; overflow: hidden;
     background: var(--overlay-bg); border: 1px solid var(--border); border-radius: var(--r-lg);
     box-shadow: var(--shadow-2);
   }
   #browserBox { width: min(620px, 100%); }
   #paletteInput {
     width: 100%; min-height: 50px; padding: 0 var(--sp-4);
     border: 0; border-bottom: 1px solid var(--border); border-radius: 0;
     background: transparent; box-shadow: none; font-size: var(--fs-lg);
   }
   #paletteList { max-height: 320px; overflow-y: auto; padding: var(--sp-1); }
   .pal-item { padding: var(--sp-2) var(--sp-3); border-radius: var(--r-sm); font-size: var(--fs-md); cursor: pointer; }
   .pal-item.sel, .pal-item:hover { background: var(--accent-soft); }
   .pal-item .palhint { color: var(--muted); font-size: var(--fs-xs); margin-left: var(--sp-2); }
   #browserPath { padding: var(--sp-3) var(--sp-4); font-size: var(--fs-sm); color: var(--muted); border-bottom: 1px solid var(--border); overflow-wrap: anywhere; }
   #browserList { max-height: 46vh; overflow-y: auto; padding: var(--sp-1); }
   .dir-item { display: flex; justify-content: space-between; gap: 10px; padding: var(--sp-2) var(--sp-3); border-radius: var(--r-sm); font-size: var(--fs-md); cursor: pointer; overflow-wrap: anywhere; }
   .dir-item:hover { background: var(--accent-soft); }
   .dir-item .repo { color: var(--success); font-size: var(--fs-xs); flex: 0 0 auto; }
   .browser-actions { padding: var(--sp-3) var(--sp-4); border-top: 1px solid var(--border); }

   /* ================================================================= tasks */
   #tasksView { flex-direction: row; }
   .task-list { flex: 0 0 360px; width: 360px; min-width: 0; padding: var(--sp-4); overflow-y: auto; border-right: 1px solid var(--border); }
   .task-monitor { flex: 1 1 auto; min-width: 0; padding: var(--sp-4); overflow-y: auto; }
   .task-card { cursor: pointer; transition: border-color .12s ease; }
   .task-card:hover { border-color: var(--btn-border); }
   .task-card.selected:hover { border-color: var(--accent); }
   #runFeed { display: flex; flex-direction: column; gap: var(--sp-2); }
   .feed-line { font-size: var(--fs-sm); color: var(--muted); font-family: var(--font-mono); overflow-wrap: anywhere; }
   .feed-answer {
     padding: 10px var(--sp-3); border: 1px solid var(--success); border-radius: var(--r-md);
     background: var(--surface); white-space: pre-wrap; line-height: 1.45; overflow-wrap: anywhere;
   }
   .feed-partial { color: var(--muted); white-space: pre-wrap; font-size: var(--fs-md); line-height: 1.45; }

   /* ================================================================= login */
   #loginOverlay {
     position: fixed; inset: 0; z-index: var(--z-login);
     display: none; align-items: center; justify-content: center;
     padding: var(--sp-4); background: var(--bg); overflow-y: auto;
   }
   body.locked #loginOverlay { display: flex; }
   #loginBox {
     width: 360px; max-width: 100%; gap: var(--sp-1);
     padding: var(--sp-5); background: var(--surface);
     border: 1px solid var(--border); border-radius: var(--r-lg); box-shadow: var(--shadow-2);
   }
   #loginBox .field-label { margin-top: var(--sp-2); }
   #loginBox .input { width: 100%; }
   .login-brand { justify-content: center; margin-bottom: var(--sp-1); }
   .login-sub { text-align: center; }
   .login-submit { width: 100%; margin-top: var(--sp-2); }
   .login-error { min-height: 18px; color: var(--error); font-size: var(--fs-sm); }
   .pw-field { position: relative; }
   .pw-field .input { padding-right: 56px; }
   .pw-reveal { position: absolute; right: 4px; top: 50%; transform: translateY(-50%); color: var(--muted); }
   .pw-reveal:hover:not(:disabled) { color: var(--fg); }
   /* When not signed in, hide the whole app so only the login box shows.
      !important because showView() puts an inline display on the composer. */
   body.locked > header,
   body.locked > #main,
   body.locked > #composer { display: none !important; }
   /* Boot state: the served HTML starts `booting` so nothing paints until
      authBoot() resolves and reveals exactly one of the app or the login box.
      Without this the app flashes on screen during the /api/auth/me fetch and
      is then covered by the login overlay. */
   body.booting > header,
   body.booting > #main,
   body.booting > #composer,
   body.booting #loginOverlay { display: none !important; }
   /* Role gating is UX only; the server enforces RBAC on every route. */
   body:not([data-role="admin"]) .admin-only { display: none !important; }

   /* ============================================================== settings */
   #settingsView { flex-direction: row; }
   .settings-rail {
     flex: 0 0 220px; width: 220px; min-width: 0;
     display: flex; flex-direction: column; gap: 2px;
     padding: var(--sp-4) var(--sp-3); overflow-y: auto;
     background: var(--panel-bg); border-right: 1px solid var(--border);
   }
   .settings-rail input[type=search], #settingsSearch { width: 100%; margin-bottom: var(--sp-3); flex: 0 0 auto; }
   .rail-items { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
   .rail-item {
     width: 100%; justify-content: flex-start; text-align: left; flex: 0 0 auto;
     background: transparent; border-color: transparent; color: var(--muted); font-weight: 600;
   }
   .rail-item:hover:not(:disabled) { background: var(--btn-hover); color: var(--fg); }
   .rail-item.active, .rail-item[aria-current="page"], .rail-item[aria-current="true"], .rail-item[aria-selected="true"], .rail-item[aria-pressed="true"] {
     background: var(--accent-soft); border-color: transparent; color: var(--fg);
     box-shadow: inset 3px 0 0 var(--accent);
   }
   .settings-main {
     flex: 1 1 auto; min-width: 0; min-height: 0;
     display: flex; flex-direction: column;
     padding: var(--sp-5) var(--sp-6) 0; overflow-y: auto; overflow-x: hidden;
   }
   .settings-main > * { width: 100%; max-width: 860px; margin-left: auto; margin-right: auto; }
   .settings-head { display: flex; flex-direction: column; gap: var(--sp-1); padding-bottom: var(--sp-4); }
   .settings-head .muted { font-size: var(--fs-md); }
   .settings-section, .panel-block { display: flex; flex-direction: column; gap: var(--sp-3); min-width: 0; }
   .settings-section { padding-bottom: var(--sp-6); }
   /* A bespoke panel (users, cluster, import...) gets the same card as a
      schema group so the two kinds of content read as one screen. */
   .panel-block { padding: var(--sp-4); background: var(--surface); border: 1px solid var(--border); border-radius: var(--r-lg); }
   .panel-block:has(> details.sgroup) { padding: 0; background: transparent; border: 0; }
   .panel-block > h3, .panel-block > h4 { margin-top: var(--sp-1); }
   details.sgroup { min-width: 0; background: var(--surface); border: 1px solid var(--border); border-radius: var(--r-lg); }
   details.sgroup > summary {
     display: flex; align-items: center; gap: var(--sp-2);
     padding: var(--sp-3) var(--sp-4); border-radius: var(--r-lg);
     list-style: none; cursor: pointer; user-select: none;
     font-size: var(--fs-md); font-weight: 700; color: var(--fg);
   }
   details.sgroup > summary::-webkit-details-marker { display: none; }
   details.sgroup > summary::before {
     content: ""; flex: 0 0 auto; width: 6px; height: 6px; margin: 0 2px;
     border-right: 2px solid var(--muted); border-bottom: 2px solid var(--muted);
     transform: rotate(-45deg); transition: transform .12s ease;
   }
   details.sgroup[open] > summary::before { transform: rotate(45deg); }
   details.sgroup > summary:hover { background: var(--btn-hover); }
   details.sgroup[open] > summary { border-bottom: 1px solid var(--border); border-radius: var(--r-lg) var(--r-lg) 0 0; }
   details.sgroup > summary:focus-visible { outline: none; box-shadow: var(--focus-ring); }
   .sbody { display: flex; flex-direction: column; gap: var(--sp-3); padding: var(--sp-4); min-width: 0; }
   .sbody > .field + .field { padding-top: var(--sp-3); border-top: 1px solid var(--border); }
   /* Field: label, control, hint. .field-row puts a switch on the right of
      its label (macOS-style), with the hint underneath across the row. */
   .field { position: relative; display: flex; flex-direction: column; gap: 6px; min-width: 0; }
   .field-label { display: flex; align-items: center; flex-wrap: wrap; gap: var(--sp-2); font-size: var(--fs-sm); font-weight: 600; color: var(--fg); }
   .field-control { display: flex; align-items: center; gap: var(--sp-2); min-width: 0; }
   .field-control > .input, .field-control > select, .field-control > textarea,
   .field-control > input:not([type=checkbox]):not([type=radio]), .field-control > .range,
   .field-control > .readonly { flex: 1 1 auto; width: 100%; }
   .field-control > input[type=number]:not(.w-full) { flex: 0 1 200px; }
   .field-control > .seg { flex-wrap: wrap; }
   .field-hint { margin: 0; font-size: var(--fs-xs); color: var(--muted); line-height: 1.45; }
   .field-row { flex-direction: row; flex-wrap: wrap; align-items: center; column-gap: var(--sp-4); }
   .field-row > .field-label { flex: 1 1 200px; }
   .field-row > .field-control { flex: 0 0 auto; margin-left: auto; }
   .field-row > .field-hint { flex: 1 0 100%; }
   .field.dirty::before {
     content: ""; position: absolute; left: calc(-1 * var(--sp-3)); top: 2px; bottom: 2px;
     width: 3px; border-radius: 2px; background: var(--accent);
   }
   .sbody > .field + .field.dirty::before { top: calc(var(--sp-3) + 2px); }
   .field.match { border-radius: var(--r-sm); }
   .settings-bar {
     position: sticky; bottom: 0; z-index: 5;
     display: flex; align-items: center; flex-wrap: wrap; gap: var(--sp-2) var(--sp-3);
     margin: auto auto 0; padding: var(--sp-3) var(--sp-4);
     background: var(--overlay-bg); border: 1px solid var(--border); border-bottom: 0;
     border-radius: var(--r-lg) var(--r-lg) 0 0; box-shadow: var(--shadow-2);
   }
   /* A zero flex-basis keeps the count on the same line as Discard / Apply
      at 360px (it ellipsises instead of pushing Apply onto a second row). */
   #settingsDirty { flex: 1 1 0; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-weight: 600; color: var(--fg); }
   .settings-restart { display: inline-flex; align-items: center; flex-wrap: wrap; gap: var(--sp-2); min-width: 0; font-size: var(--fs-sm); color: var(--muted); }
   .settings-section-title { margin-top: var(--sp-2); color: var(--muted); font-size: var(--fs-sm); text-transform: uppercase; letter-spacing: .04em; }
   .sub-panel { display: flex; flex-direction: column; gap: var(--sp-3); min-width: 0; }
   /* "Advanced" folds inside a group: a quiet disclosure, not another card. */
   details.sgroup-advanced { border-top: 1px solid var(--border); padding-top: var(--sp-3); }
   /* A group whose fields are all advanced (Benchmark) has no rule to separate. */
   details.sgroup-advanced:first-child { border-top: 0; padding-top: 0; }
   details.sgroup-advanced > summary {
     display: inline-flex; align-items: center; gap: var(--sp-2);
     list-style: none; cursor: pointer; user-select: none;
     font-size: var(--fs-sm); font-weight: 600; color: var(--muted);
   }
   details.sgroup-advanced > summary::-webkit-details-marker { display: none; }
   details.sgroup-advanced > summary::before {
     content: ""; width: 5px; height: 5px; margin: 0 2px;
     border-right: 2px solid currentColor; border-bottom: 2px solid currentColor;
     transform: rotate(-45deg); transition: transform .12s ease;
   }
   details.sgroup-advanced[open] > summary::before { transform: rotate(45deg); }
   details.sgroup-advanced > summary:hover { color: var(--fg); }
   details.sgroup-advanced > .sbody { padding: var(--sp-3) 0 0; }
   .list-row { padding: var(--sp-2) 0; border-bottom: 1px solid var(--border); flex-wrap: nowrap; }
   .list-row:last-child { border-bottom: 0; }
   .scope-list { max-height: 260px; overflow-y: auto; }
   /* Prefill curve: one bar per agent step, colours from the chart tokens. */
   .chart { display: flex; flex-direction: column; gap: var(--sp-1); min-width: 0; }
   .curve-row { display: grid; grid-template-columns: 3.5em minmax(0, 1fr) 6ch; align-items: center; gap: var(--sp-2); font-size: var(--fs-xs); }
   .curve-track { height: 8px; background: var(--meter-bg); border-radius: var(--r-pill); overflow: hidden; }
   .curve-fill { display: block; height: 100%; width: var(--fill, 0%); background: var(--chart-1); border-radius: inherit; }
   .curve-fill.growing { background: var(--warn); }
   .curve-value { text-align: right; }
   .curve-title { margin-bottom: var(--sp-1); }
   .curve-verdict { margin-top: var(--sp-1); }

   /* ========================================================= admin panels */
   #dropZone {
     padding: 18px; border: 1px dashed var(--btn-border); border-radius: var(--r-md);
     text-align: center; color: var(--muted); font-size: var(--fs-md); cursor: pointer;
     transition: border-color .15s ease, background-color .15s ease;
   }
   #dropZone:hover { border-color: var(--accent); }
   #dropZone.dragging { border-color: var(--accent); background: var(--accent-soft); }
   .agent-card { display: flex; flex-direction: column; gap: var(--sp-2); padding: var(--sp-3); background: var(--surface); border: 1px solid var(--border); border-radius: var(--r-lg); }
   .agent-head { display: flex; align-items: center; gap: 10px; min-width: 0; }
   .agent-name { flex: 1 1 auto; font-weight: 600; }
   .agent-desc { width: 100%; }
   .agent-merged { padding: 10px var(--sp-3); border: 1px solid var(--accent); border-radius: var(--r-md); }
   .agent-answer { font-size: var(--fs-md); margin-top: 6px; white-space: pre-wrap; overflow-wrap: anywhere; }
   .agent-result { padding: var(--sp-2) 10px; border: 1px solid var(--border); border-radius: var(--r-md); }
   .agent-result summary { cursor: pointer; font-size: var(--fs-sm); color: var(--muted); }
   .import-detail { overflow-wrap: anywhere; }

   /* ==================================================== panels-ops: charts */
   /* uiChart (ui_script_charts.py). Series colour follows position: s1..s4 ->
      --chart-1..4 through --series, so marks follow light/dark with the
      tokens. Text never wears the series colour: values and labels stay in
      --fg / --muted, a coloured mark beside them carries identity. */
   .uichart { display: flex; flex-direction: column; gap: var(--sp-2); min-width: 0; font-size: var(--fs-xs); color: var(--muted); }
   .uichart .s1 { --series: var(--chart-1); }
   .uichart .s2 { --series: var(--chart-2); }
   .uichart .s3 { --series: var(--chart-3); }
   .uichart .s4 { --series: var(--chart-4); }
   .uichart-empty { margin: 0; padding: var(--sp-3) 0; text-align: center; }
   .uichart-legend { display: flex; flex-wrap: wrap; gap: var(--sp-1) var(--sp-3); }
   .uichart-key { display: inline-flex; align-items: center; gap: var(--sp-1); color: var(--fg); }
   .uichart-key::before { content: ""; width: 12px; height: 4px; border-radius: var(--r-pill); background: var(--series); }
   .uichart-rows { display: flex; flex-direction: column; gap: var(--sp-2); }
   .uichart-group { display: grid; grid-template-columns: minmax(0, 12em) minmax(0, 1fr) 9ch; align-items: center; gap: 2px var(--sp-2); }
   .uichart-label { color: var(--fg); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
   .uichart-track { display: block; height: 8px; background: var(--meter-bg); border-radius: var(--r-pill); overflow: hidden; }
   .uichart-fill { display: block; height: 100%; width: var(--fill, 0%); background: var(--series); border-radius: inherit; transition: width .3s ease; }
   .uichart-value { color: var(--fg); text-align: right; white-space: nowrap; }
   .uichart-plot { display: grid; grid-template-columns: auto minmax(0, 1fr); gap: var(--sp-1) var(--sp-2); }
   .uichart-yaxis { display: flex; flex-direction: column; justify-content: space-between; text-align: right; }
   .uichart-xaxis { display: flex; justify-content: space-between; gap: var(--sp-2); }
   .uichart-svg { display: block; width: 100%; overflow: visible; }
   .uichart-line { fill: none; stroke: var(--series); stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; vector-effect: non-scaling-stroke; }
   .uichart-area { fill: var(--series); fill-opacity: .14; stroke: none; }
   .uichart-grid { stroke: var(--border); stroke-width: 1; vector-effect: non-scaling-stroke; }
   .uichart-hit { fill: transparent; }
   .uichart-hit:hover { fill: color-mix(in srgb, var(--fg) 6%, transparent); }
   .uichart-sparkline { flex-direction: row; align-items: center; }
   .uichart-sparkline .uichart-svg { flex: 1 1 auto; min-width: 0; }
   @media (max-width: 480px) {
     .uichart-group { grid-template-columns: minmax(0, 7em) minmax(0, 1fr) 8ch; }
   }
   /* Ops panels (ui_script_ops.py): a meter with its figure under it inside a
      table cell, and a one-time secret (the pairing code) set large. */
   .ops-meter { display: flex; flex-direction: column; gap: 2px; min-width: 72px; }
   .ops-meter .meter > div.warn { background: var(--warn); }
   .ops-meter .meter > div.error { background: var(--error); }
   .ops-code { font-family: var(--font-mono); font-size: var(--fs-xl); letter-spacing: .12em; }
   /* The cluster node table has six columns; on a phone it scrolls inside
      the card instead of widening the page. */
   .ops-scroll { overflow-x: auto; min-width: 0; }
   /* ============================================== panels-ops: charts end */

   /* ============================================================ dashboards */
   /* Admin > Dashboards (ui_script_dashboards.py). KPI tiles: a label, one
      figure in tabular numerals, one line of context; the grid reflows from
      six across to two on a phone. The chart grid does the same for uiChart
      hosts. The A/B card shows two answers side by side (stacked on a phone),
      each scrolling on its own so a long answer cannot push the vote buttons
      off screen. */
   .dash-kpis { display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: var(--sp-2); }
   .dash-kpi { display: flex; flex-direction: column; gap: 2px; min-width: 0; padding: var(--sp-2) var(--sp-3); background: var(--surface-2); border: 1px solid var(--border); border-radius: var(--r-md); }
   .dash-kpi-label { font-size: var(--fs-xs); color: var(--muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
   .dash-kpi-value { font-size: var(--fs-xl); font-weight: 600; color: var(--fg); font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
   .dash-kpi-sub { font-size: var(--fs-xs); color: var(--muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
   .dash-charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: var(--sp-4) var(--sp-3); }
   .dash-chart { display: flex; flex-direction: column; gap: var(--sp-1); min-width: 0; }
   .dash-ab { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: var(--sp-3); }
   .dash-answer { max-height: 360px; overflow-y: auto; padding: var(--sp-3); border: 1px solid var(--border); border-radius: var(--r-md); font-size: var(--fs-md); white-space: pre-wrap; overflow-wrap: anywhere; }
   @media (max-width: 640px) {
     .dash-ab { grid-template-columns: minmax(0, 1fr); }
   }
   /* ========================================================= dashboards end */

   /* ================================================ chat surface additions */
   /* Owned by the chat surface (ui_script_chat.py): the composer's pickers,
      attachment chips and mic, thumbnails in a question, the inline decision
      card (tool approval and cloud consent share it), replayed reasoning and
      metadata on a stored answer, and the header notifications bell. Built
      from the kit's tokens; the card, pill, badge, seg and menu are the kit's. */
   .composer-select { min-height: var(--ctl-h-sm); max-width: 168px; padding-left: var(--sp-2); font-size: var(--fs-sm); border-radius: var(--r-sm); }
   .composer-box.dragging { border-color: var(--accent); box-shadow: var(--focus-ring); }
   .att-chips { display: flex; flex-wrap: wrap; gap: var(--sp-1); min-width: 0; padding: 0 var(--sp-3) var(--sp-1); }
   .att-chip {
     display: inline-flex; align-items: center; gap: var(--sp-1);
     max-width: 240px; min-width: 0; padding: 2px var(--sp-2);
     background: var(--chip-bg); border: 1px solid var(--border); border-radius: var(--r-pill);
     color: var(--fg); font-size: var(--fs-xs); line-height: 1.5;
     overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
   }
   .att-chips .att-chip { padding-right: 2px; }
   .att-chip.uploading { opacity: 0.7; }
   .att-chip .att-name { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
   .att-chip .att-remove { width: 20px; min-width: 20px; min-height: 20px; padding: 0; border-radius: 50%; font-size: var(--fs-sm); }
   .att-thumb { flex: 0 0 auto; width: 20px; height: 20px; object-fit: cover; border-radius: var(--r-sm); }
   .mic-btn.recording { color: var(--error); border-color: var(--error); }
   .msg-atts { display: flex; flex-wrap: wrap; gap: var(--sp-1); margin-top: var(--sp-2); white-space: normal; }
   .msg-thumb { display: block; max-width: 180px; max-height: 140px; object-fit: cover; border-radius: var(--r-md); border: 1px solid var(--border); background: var(--surface); cursor: zoom-in; }
   .msg.user .att-chip { background: color-mix(in srgb, var(--accent-fg) 14%, transparent); border-color: transparent; color: inherit; }
   .msg-reasoning { margin: 0 0 var(--sp-2); }
   .msg-reasoning > summary { cursor: pointer; user-select: none; color: var(--muted); font-family: var(--font-mono); font-size: var(--fs-xs); }
   .msg-reasoning-body { margin: var(--sp-1) 0 0; max-height: 200px; overflow: auto; color: var(--muted); font-style: italic; font-size: var(--fs-sm); white-space: pre-wrap; overflow-wrap: anywhere; }
   .msg-meta { display: flex; flex-wrap: wrap; align-items: center; gap: var(--sp-2); margin-top: var(--sp-1); color: var(--muted); font-size: var(--fs-xs); white-space: normal; }
   .gnode.tool.awaiting .gpill { background: color-mix(in srgb, var(--warn) 16%, transparent); color: var(--warn); }
   .gnode.decision::before { border-color: var(--warn); background: var(--warn); }
   .gnode.decision.resolved::before { border-color: var(--border); background: var(--surface); }
   .decision-card { gap: var(--sp-2); padding: var(--sp-3); border-color: color-mix(in srgb, var(--warn) 45%, var(--border)); }
   .decision.resolved .decision-card { border-color: var(--border); }
   .decision-card .decision-timer { margin-left: auto; font-size: var(--fs-xs); }
   /* "Edit arguments" sits under the arguments, not stretched across the card. */
   .decision-card > .btn-ghost { align-self: flex-start; }
   .decision-args { margin: 0; max-height: 180px; overflow: auto; padding: var(--sp-2); background: var(--surface-2); border-radius: var(--r-sm); font-size: var(--fs-xs); white-space: pre-wrap; overflow-wrap: anywhere; }
   .decision-edit { min-height: 120px; font-family: var(--font-mono); font-size: var(--fs-xs); }
   .decision-actions { align-items: center; gap: var(--sp-2); }
   .decision-pending { align-self: stretch; display: flex; flex-direction: column; gap: var(--sp-2); }
   .notif { position: relative; }
   .notif-btn { position: relative; }
   .notif-badge {
     position: absolute; top: 1px; right: 1px;
     min-width: 16px; justify-content: center; padding: 0 4px;
     background: var(--accent); color: var(--accent-fg); border-color: transparent; border-radius: var(--r-pill);
     font-size: 10px; line-height: 16px; pointer-events: none;
   }
   .notif-pop { right: 0; top: calc(100% + 6px); width: min(360px, calc(100vw - 2 * var(--sp-4))); max-width: none; max-height: min(480px, 70vh); overflow-y: auto; }
   /* On a phone the bell is not at the header's right edge, so a popover
      right-aligned to it ran off the left of the screen: pin it to the viewport. */
   @media (max-width: 480px) {
     .notif-pop { position: fixed; top: calc(var(--header-h) + var(--sp-1)); left: var(--sp-3); right: var(--sp-3); width: auto; }
   }
   .notif-head { padding: var(--sp-1) var(--sp-2); }
   .notif-list { gap: 2px; }
   .notif-item {
     display: flex; flex-direction: column; align-items: flex-start; gap: 2px;
     width: 100%; min-height: 0; padding: var(--sp-2);
     background: transparent; border-color: transparent; border-radius: var(--r-sm);
     text-align: left; white-space: normal; font-weight: 400;
   }
   .notif-item .notif-title { font-weight: 600; font-size: var(--fs-sm); }
   .notif-item .notif-body { color: var(--muted); font-size: var(--fs-sm); overflow-wrap: anywhere; }
   .notif-item .notif-when { color: var(--muted); font-size: var(--fs-xs); font-family: var(--font-mono); }
   .notif-item.unread .notif-title::before, .notif-item.unread .notif-body:first-child::before {
     content: ""; display: inline-block; width: 7px; height: 7px; margin-right: var(--sp-1);
     border-radius: 50%; background: var(--accent); vertical-align: middle;
   }
   .notif-item.error .notif-title { color: var(--error); }
   .notif-item.warn .notif-title, .notif-item.warning .notif-title { color: var(--warn); }
   /* ============================================ chat surface additions end */

   /* ============================== integrator (Wave 4a): narrow-screen tables */
   /* The settings scroller clips sideways overflow (overflow-x: hidden), so on a
      phone a wide table (tool catalogue, engines, audit) lost its right-hand
      columns, and overflow-wrap:anywhere squeezed the first column to a letter
      per line. Under 640px a table scrolls inside its own box instead, and the
      first column keeps a readable minimum. */
   @media (max-width: 640px) {
     .settings-main .table, .settings-main table.models { display: block; overflow-x: auto; }
     .settings-main .table td:first-child, .settings-main table.models td:first-child { min-width: 9em; }
   }
   /* ========================== integrator (Wave 4a): narrow-screen tables end */

   /* ============================================================= approvals */
   /* The header's Approvals control (ui_markup.py; apprRender in
      ui_script_settings.py). It must stay on screen at every width, so it
      sheds words instead of disappearing: "Approvals: Ask every action" on a
      wide window, the mode alone below 1100px, a short label below 760px and
      the shield alone below 640px. The shield's glyph and colour still say
      the mode there (check = Ask every action, split = Ask for changes,
      slash = Don't ask) and its tooltip and aria-label carry the words. A
      small lock marks a mode the administrator's minimum sets. */
   .appr { position: relative; flex: 0 0 auto; }
   .appr-btn { gap: 6px; padding: 0 var(--sp-2); font-size: var(--fs-sm); }
   .appr-btn .appr-prefix { color: var(--muted); font-weight: 400; }
   .appr-btn .appr-label, .appr-btn .appr-short { font-weight: 600; }
   .appr-btn .appr-short, .appr-btn .appr-lock { display: none; }
   .appr-btn .caret { color: var(--muted); font-size: 10px; }
   .appr-icon .appr-g { display: none; }
   .appr-btn[data-mode="all"] .appr-g-all,
   .appr-btn[data-mode="risky"] .appr-g-risky,
   .appr-btn[data-mode="off"] .appr-g-off { display: inline; }
   .appr-btn[data-mode="all"] .appr-icon { color: var(--success); }
   .appr-btn[data-mode="risky"] .appr-icon { color: var(--accent); }
   .appr-btn[data-mode="off"] .appr-icon,
   .appr-btn[data-mode="off"] .appr-label,
   .appr-btn[data-mode="off"] .appr-short { color: var(--warn); }
   .appr-btn.locked .appr-lock { display: block; color: var(--muted); }
   .appr-pop { right: 0; top: calc(100% + 6px); width: min(340px, calc(100vw - 2 * var(--sp-4))); max-width: none; }
   .appr-head { padding: var(--sp-1) var(--sp-2) var(--sp-2); }
   .appr-head .hint { margin-top: 2px; }
   .menu > .appr-opt {
     flex-direction: column; align-items: flex-start; gap: 2px;
     min-height: 0; padding: var(--sp-2); white-space: normal; text-align: left;
   }
   .appr-opt .appr-opt-title { font-weight: 600; }
   .appr-opt .appr-opt-help { color: var(--muted); font-size: var(--fs-xs); font-weight: 400; line-height: 1.4; }
   .menu > .appr-opt.active { background: var(--accent-soft); border-color: var(--accent); }
   .menu > .appr-opt:disabled { cursor: not-allowed; }
   .appr-locknote, .appr-unattended { padding: var(--sp-1) var(--sp-2) 0; }
   @media (max-width: 1100px) {
     .appr-btn .appr-prefix { display: none; }
   }
   @media (max-width: 760px) {
     .appr-btn .appr-label { display: none; }
     .appr-btn .appr-short { display: inline; }
   }
   @media (max-width: 640px) {
     .appr-btn .appr-short, .appr-btn .caret { display: none; }
     .appr-btn { position: relative; width: var(--ctl-h); min-width: var(--ctl-h); padding: 0; }
     .appr-btn.locked .appr-lock { position: absolute; right: 1px; bottom: 3px; }
   }
   /* On a phone the control is not at the header's right edge, so pin its
      menu to the viewport like the notifications popover. */
   @media (max-width: 480px) {
     .appr-pop { position: fixed; top: calc(var(--header-h) + var(--sp-1)); left: var(--sp-3); right: var(--sp-3); width: auto; }
   }
   /* ========================================================= approvals end */

   /* ============================================================ responsive */
   @media (max-width: 900px) {
     .brand .build { display: none; }
   }
   /* Below ~940px the 820px column's gutter is narrower than #chatMenuBtn, so the
      first right-aligned user bubble sat under the floating button. Start the
      transcript below it instead. */
   @media (max-width: 940px) {
     #chat { padding-top: calc(var(--sp-2) * 2 + var(--ctl-h)); }
   }
   @media (max-width: 760px) {
     /* On a phone the status pill truncates to "Model: st..." and costs a
        whole header slot; the same numbers are in Settings. */
     #status { display: none; }
     body > header { padding: var(--sp-2) var(--sp-3); gap: var(--sp-2); }
     body > #composer { padding: var(--sp-2); }
     .msg { max-width: 92%; }
     /* Settings stacks: the rail becomes a horizontally scrolling segmented
        strip above the content. */
     #settingsView { flex-direction: column; }
     .settings-rail {
       flex: 0 0 auto; width: auto; gap: var(--sp-2);
       padding: var(--sp-2) var(--sp-3); overflow: visible;
       border-right: 0; border-bottom: 1px solid var(--border);
     }
     .settings-rail input[type=search], #settingsSearch { margin: 0; }
     .rail-items {
       flex-direction: row; overflow-x: auto; scrollbar-width: none;
       padding: 2px; background: var(--surface-2); border: 1px solid var(--border); border-radius: var(--r-md);
     }
     .rail-items::-webkit-scrollbar { display: none; }
     .rail-item { width: auto; min-height: calc(var(--ctl-h) - 6px); padding: 0 var(--sp-3); font-size: var(--fs-sm); border-radius: calc(var(--r-md) - 3px); }
     .rail-item.active, .rail-item[aria-current="page"], .rail-item[aria-current="true"], .rail-item[aria-selected="true"], .rail-item[aria-pressed="true"] {
       background: var(--btn-bg); border-color: var(--border); box-shadow: var(--shadow-1);
     }
     .settings-main { padding: var(--sp-4) var(--sp-3) 0; }
     .settings-bar { padding: var(--sp-2) var(--sp-3); }
     /* Tasks: list above the monitor, one scroll for both. */
     #tasksView { flex-direction: column; overflow-y: auto; }
     .task-list { flex: 0 0 auto; width: auto; overflow: visible; border-right: 0; border-bottom: 1px solid var(--border); }
     .task-monitor { overflow: visible; }
     /* The prompt library floats over the chat instead of squeezing it. */
     #promptsPanel {
       position: fixed; top: 0; right: 0; bottom: 0; z-index: var(--z-pop);
       width: min(340px, 100%); box-shadow: var(--shadow-2);
     }
   }
   @media (max-width: 480px) {
     .brand strong, .user-chip { display: none; }
     body > header .seg > button { padding: 0 var(--sp-2); }
     body > header .seg { overflow-x: auto; scrollbar-width: none; }
     #ctxHint, .composer-meta { display: none; }
     .composer-toolbar .seg > button { padding: 0 var(--sp-2); }
     #chat { padding-left: var(--sp-3); padding-right: var(--sp-3); }
     /* The one-row composer clipped the wrapped placeholder mid-word on phones. */
     #input::placeholder { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
     .field-row > .field-label { flex-basis: 140px; }
     .toast-host { right: var(--sp-3); width: calc(100vw - 2 * var(--sp-3)); }
   }
   @media (prefers-reduced-motion: reduce) {
     *, *::before, *::after { transition-duration: 0s !important; }
     .toast { animation: none; }
   }

   /* ======================================================= state overrides */
   /* Every id rule above that sets `display` is matched here, because an id
      selector outranks the plain `.hidden` below. */
   #palette.hidden, #browser.hidden, #paletteBox.hidden, #browserBox.hidden, #promptsPanel.hidden, #runFeed.hidden, #loginOverlay.hidden, #main.hidden, #chat.hidden, #composer.hidden { display: none; }
   /* The hidden attribute must win over component display rules too
      (.panel-block, .row, .card all set display). */
   [hidden] { display: none !important; }
   /* The catch-all for the `hidden` state class, LAST so it wins over any
      earlier same-specificity `display`. Without it a `.hidden` element with
      no id-qualified rule of its own stays on screen: that is exactly how the
      account popup once sat permanently open over the header. */
   .hidden { display: none; }
 """

__all__ = ["UI_CSS"]
