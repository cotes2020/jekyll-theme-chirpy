// @ts-nocheck
(function () {
    Mdd.injectCSS('chatbot', `
        .cb-outer { display:flex; flex-direction:column; flex:1; min-height:0; min-width:0; width:100%; position:relative; }
        .cb-layout { display:flex; gap:14px; flex:1; min-height:0; align-items:stretch; }

        /* 좌·우 사이드바 공통 */
        .cb-sidebar { display:flex; flex-direction:column; min-height:0; border:1px solid var(--border); background:var(--bg-secondary); border-radius:var(--radius-lg); overflow:hidden; box-shadow:0 1px 0 rgba(0,0,0,0.04); }
        .cb-sidebar-left { width:220px; flex-shrink:0; }
        .cb-sidebar-right { width:292px; flex-shrink:0; }
        .cb-panel-heading { font-size:var(--font-size-2xs); font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:var(--text-tertiary); margin:0 0 10px; }
        .cb-sidebar-header { padding:14px 14px 12px; border-bottom:1px solid var(--border); flex-shrink:0; }
        .cb-sidebar-header .cb-panel-heading { margin-bottom:8px; }

        .cb-api-row { display:flex; gap:6px; align-items:center; }
        .cb-api-row input { flex:1; font-size:var(--font-size-xs); padding:6px 8px; }
        .cb-key-status { font-size:var(--font-size-2xs); color:var(--text-tertiary); margin-top:4px; }

        .cb-model-label { font-size:var(--font-size-xs); font-weight:500; color:var(--text-secondary); margin:12px 0 4px; display:block; }

        .cb-options { padding:12px 14px; flex-shrink:0; border-bottom:1px solid var(--border); }
        .cb-option-row { display:flex; align-items:center; justify-content:space-between; margin-bottom:8px; }
        .cb-option-row label { font-size:var(--font-size-xs); color:var(--text-secondary); }
        .cb-toggle { position:relative; width:36px; height:20px; }
        .cb-toggle input { opacity:0; width:0; height:0; }
        .cb-toggle-slider {
            position:absolute; inset:0; background:var(--bg-active); border-radius:10px;
            cursor:pointer; transition:background 0.2s;
        }
        .cb-toggle-slider::before {
            content:''; position:absolute; width:16px; height:16px; border-radius:50%;
            background:#fff; left:2px; top:2px; transition:transform 0.2s;
        }
        .cb-toggle input:checked + .cb-toggle-slider { background:var(--accent); }
        .cb-toggle input:checked + .cb-toggle-slider::before { transform:translateX(16px); }

        /* 시스템 프롬프트 (우측 패널 하단) */
        .cb-sysprompt { flex:1; min-height:120px; padding:12px 14px; overflow-y:auto; display:flex; flex-direction:column; border-top:1px solid var(--border); }
        .cb-sysprompt label { font-size:var(--font-size-xs); font-weight:500; color:var(--text-secondary); margin-bottom:4px; display:block; }
        .cb-sysprompt textarea {
            flex:1; min-height:120px; font-size:var(--font-size-xs); line-height:1.5;
            resize:none; padding:8px; background:var(--bg-tertiary);
        }
        .cb-sysprompt textarea:read-only { opacity:0.72; cursor:default; }

        /* 중앙: 좁은 채팅 열 */
        .cb-chat-stage { flex:1; min-width:0; display:flex; justify-content:center; align-items:stretch; padding:0 4px; }
        .cb-chat {
            flex:0 1 540px; width:100%; max-width:540px;
            display:flex; flex-direction:column;
            background:var(--bg-primary); border:1px solid var(--border);
            border-radius:var(--radius-lg); overflow:hidden;
            box-shadow:0 2px 12px rgba(0,0,0,0.06);
        }

        .cb-chat-header {
            padding:10px 14px; border-bottom:1px solid var(--border);
            display:flex; align-items:center; justify-content:space-between; gap:8px;
            background:var(--bg-secondary); flex-shrink:0;
        }
        .cb-chat-header-title { font-size:var(--font-size-sm); font-weight:600; color:var(--text-primary); min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
        .cb-chat-header-actions { display:flex; gap:4px; flex-wrap:wrap; justify-content:flex-end; }

        .cb-messages {
            flex:1; min-height:0; overflow-y:auto; padding:14px; display:flex;
            flex-direction:column; gap:10px;
        }

        .cb-msg { max-width:80%; padding:10px 14px; border-radius:14px; font-size:var(--font-size-sm); line-height:1.6; animation:fadeIn 0.3s ease; word-break:break-word; }
        .cb-msg-user {
            align-self:flex-end; background:var(--accent); color:#fff;
            border-bottom-right-radius:4px; white-space:pre-wrap;
        }
        .cb-msg-bot {
            align-self:flex-start; background:var(--bg-tertiary); color:var(--text-primary);
            border:1px solid var(--border); border-bottom-left-radius:4px;
        }
        .cb-msg-error { background:var(--error-subtle) !important; border-color:var(--error) !important; color:var(--error) !important; }

        /* 마크다운 렌더링 */
        .cb-msg-bot p { margin:0 0 8px; } .cb-msg-bot p:last-child { margin:0; }
        .cb-msg-bot strong { font-weight:700; }
        .cb-msg-bot em { font-style:italic; }
        .cb-msg-bot code { background:rgba(255,255,255,0.08); padding:1px 5px; border-radius:3px; font-family:'Cascadia Code','Consolas',monospace; font-size:var(--font-size-xs); }
        .cb-msg-bot pre { border:1px solid var(--border); border-radius:var(--radius-sm); padding:0; margin:8px 0; overflow-x:auto; position:relative; }
        .cb-msg-bot pre code { background:none !important; padding:10px 12px; font-size:var(--font-size-xs); line-height:1.5; display:block; }
        .cb-code-header { display:flex; justify-content:space-between; align-items:center; padding:4px 8px 4px 12px; background:rgba(255,255,255,0.04); border-bottom:1px solid var(--border); font-size:var(--font-size-2xs); }
        .cb-code-lang { color:var(--text-tertiary); font-weight:600; text-transform:uppercase; letter-spacing:0.5px; }
        .cb-msg-bot ul, .cb-msg-bot ol { margin:6px 0; padding-left:20px; }
        .cb-msg-bot li { margin:2px 0; }
        .cb-msg-bot a { color:var(--accent); text-decoration:underline; }
        .cb-msg-bot blockquote { border-left:3px solid var(--accent); padding-left:10px; margin:6px 0; color:var(--text-secondary); }
        .cb-msg-bot hr { border:none; border-top:1px solid var(--border); margin:8px 0; }
        .cb-msg-bot table { border-collapse:collapse; margin:8px 0; font-size:var(--font-size-xs); width:100%; }
        .cb-msg-bot th, .cb-msg-bot td { border:1px solid var(--border); padding:4px 8px; text-align:left; }
        .cb-msg-bot th { background:var(--bg-primary); font-weight:600; }
        .cb-msg-bot tr:nth-child(even) td { background:rgba(255,255,255,0.02); }

        .cb-cursor-blink { animation:cbCursorBlink 0.8s step-end infinite; color:var(--accent); }
        @keyframes cbCursorBlink { 0%,100%{opacity:1} 50%{opacity:0} }

        .cb-stop-btn { background:var(--error); color:#fff; border:none; border-radius:var(--radius-md); cursor:pointer; font-size:var(--font-size-xs); padding:6px 14px; font-family:inherit; font-weight:600; }

        .cb-typing span {
            display:inline-block; width:5px; height:5px; background:var(--text-tertiary);
            border-radius:50%; animation:cbTyping 1.4s infinite ease-in-out both; margin:0 1px;
        }
        .cb-typing span:nth-child(1) { animation-delay:-0.32s; }
        .cb-typing span:nth-child(2) { animation-delay:-0.16s; }
        @keyframes cbTyping { 0%,80%,100%{transform:scale(0)} 40%{transform:scale(1)} }

        /* 입력 영역 */
        .cb-input-area { padding:12px 16px; border-top:1px solid var(--border); background:var(--bg-secondary); }
        .cb-input-row { display:flex; gap:8px; }
        .cb-input-row textarea { flex:1; min-height:40px; max-height:80px; resize:none; font-size:var(--font-size-sm); padding:8px 12px; }
        .cb-send-btn {
            width:40px; background:var(--accent); color:#fff; border:none;
            border-radius:var(--radius-md); cursor:pointer; font-size:16px;
            display:flex; align-items:center; justify-content:center;
            transition:background var(--transition);
        }
        .cb-send-btn:hover { background:var(--accent-hover); }
        .cb-token-bar { display:flex; justify-content:space-between; margin-top:6px; font-size:var(--font-size-2xs); color:var(--text-tertiary); font-family:monospace; }

        .cb-mic-btn {
            width:40px; background:transparent; color:var(--text-secondary); border:1px solid var(--border);
            border-radius:var(--radius-md); cursor:pointer; font-size:16px;
            display:flex; align-items:center; justify-content:center; transition:all var(--transition);
        }
        .cb-mic-btn:hover { border-color:var(--accent); color:var(--accent); }
        .cb-mic-btn.recording { background:var(--error); color:#fff; border-color:var(--error); animation:cbMicPulse 1s infinite; }
        @keyframes cbMicPulse { 0%,100%{opacity:1} 50%{opacity:0.6} }

        .cb-attach-area { display:flex; gap:6px; align-items:center; margin-bottom:4px; flex-wrap:wrap; }
        .cb-attach-thumb { width:48px; height:48px; object-fit:cover; border-radius:var(--radius-sm); border:1px solid var(--border); position:relative; }
        .cb-attach-wrap { position:relative; display:inline-block; }
        .cb-attach-remove { position:absolute; top:-4px; right:-4px; width:16px; height:16px; border-radius:50%; background:var(--error); color:#fff; border:none; font-size:var(--font-size-2xs); line-height:16px; text-align:center; cursor:pointer; padding:0; }
        .cb-attach-btn { background:none; border:1px dashed var(--border); border-radius:var(--radius-sm); width:40px; height:40px; font-size:18px; cursor:pointer; color:var(--text-tertiary); display:flex; align-items:center; justify-content:center; }
        .cb-attach-btn:hover { border-color:var(--accent); color:var(--accent); }
        .cb-input-area.drag-over { outline:2px dashed var(--accent); outline-offset:-2px; border-radius:var(--radius-md); }

        .cb-session-bar { display:flex; gap:2px; padding:6px 16px; border-bottom:1px solid var(--border); background:var(--bg-secondary); overflow-x:auto; flex-shrink:0; }
        .cb-session-tab { display:flex; align-items:center; gap:4px; padding:4px 10px; font-size:var(--font-size-xs); border:1px solid var(--border); border-radius:var(--radius-sm); background:var(--bg-tertiary); color:var(--text-secondary); cursor:pointer; white-space:nowrap; font-family:inherit; transition:all var(--transition); }
        .cb-session-tab:hover { border-color:var(--accent); color:var(--text-primary); }
        .cb-session-tab.active { background:var(--accent); color:#fff; border-color:var(--accent); }
        .cb-session-tab-name { max-width:80px; overflow:hidden; text-overflow:ellipsis; cursor:default; }
        .cb-session-tab-edit { border:1px solid var(--accent); border-radius:2px; font-size:var(--font-size-xs); padding:0 4px; width:70px; font-family:inherit; outline:none; background:var(--bg-primary); color:var(--text-primary); }
        .cb-session-tab-del { font-size:var(--font-size-xs); opacity:0.6; cursor:pointer; margin-left:2px; }
        .cb-session-tab-del:hover { opacity:1; color:#ff6b6b; }
        .cb-session-tab.active .cb-session-tab-del:hover { color:#ffd; }
        .cb-session-add { font-size:14px; font-weight:700; padding:4px 8px; color:var(--text-tertiary); }

        .cb-search-bar {
            display:none; padding:8px 16px; border-bottom:1px solid var(--border); background:var(--bg-secondary);
        }
        .cb-search-bar.open { display:flex; gap:6px; align-items:center; }
        .cb-search-bar input {
            flex:1; font-size:var(--font-size-xs); padding:6px 10px; border:1px solid var(--border); border-radius:var(--radius-sm);
            background:var(--bg-primary); color:var(--text-primary); outline:none; font-family:inherit;
        }
        .cb-search-bar input:focus { border-color:var(--accent); }
        .cb-search-bar .cb-search-nav { font-size:var(--font-size-xs); color:var(--text-tertiary); white-space:nowrap; }
        .cb-search-highlight { background:rgba(255,200,0,0.35); border-radius:2px; padding:0 1px; }
        .cb-search-highlight.current { background:rgba(255,200,0,0.7); outline:2px solid var(--accent); }

        .cb-shortcuts-overlay {
            display:none; position:absolute; top:0; left:0; right:0; bottom:0;
            background:rgba(0,0,0,0.6); z-index:100; align-items:center; justify-content:center;
        }
        .cb-shortcuts-overlay.open { display:flex; }
        .cb-shortcuts-panel {
            background:var(--bg-secondary); border:1px solid var(--border); border-radius:var(--radius-lg);
            padding:20px 24px; max-width:340px; width:90%;
        }
        .cb-shortcuts-panel h3 { margin:0 0 12px; font-size:14px; }
        .cb-shortcut-row { display:flex; justify-content:space-between; align-items:center; padding:4px 0; font-size:var(--font-size-xs); }
        .cb-shortcut-key {
            display:inline-block; padding:2px 8px; border-radius:4px; font-size:var(--font-size-xs); font-family:monospace;
            background:var(--bg-primary); border:1px solid var(--border); color:var(--text-secondary); min-width:24px; text-align:center;
        }

        .cb-msg-wrap { position:relative; }
        .cb-msg-copy { position:absolute; top:6px; right:6px; opacity:0; transition:opacity 0.2s; }
        .cb-msg-wrap:hover .cb-msg-copy { opacity:1; }

        .cb-character-block { flex:0 0 auto; padding:12px 14px; }
        .cb-character-block h4 { font-size:var(--font-size-xs); font-weight:600; margin:0 0 10px; color:var(--text-primary); }
        .cb-char-profile-wrap { display:flex; flex-direction:column; align-items:center; gap:8px; margin-top:10px; }
        .cb-char-profile-btn {
            width:72px; height:72px; border-radius:50%; border:2px solid var(--border); background:var(--bg-tertiary);
            padding:0; cursor:pointer; overflow:hidden; display:flex; align-items:center; justify-content:center; flex-shrink:0;
        }
        .cb-char-profile-btn:hover { border-color:var(--accent); }
        .cb-char-profile-btn:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
        .cb-char-profile-avatar { width:100%; height:100%; object-fit:cover; }
        .cb-char-profile-placeholder { font-size:32px; line-height:1; user-select:none; }
        .cb-char-profile-name {
            margin:0; font-size:var(--font-size-xs); font-weight:600; color:var(--text-secondary);
            text-align:center; max-width:100%; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
        }
        .cb-char-modal-body label.cb-mini { font-size:var(--font-size-2xs); color:var(--text-tertiary); display:block; margin-top:8px; }
        .cb-char-modal-body label.cb-mini:first-child { margin-top:0; }
        .cb-char-modal-body input[type="text"], .cb-char-modal-body textarea, .cb-char-modal-body select {
            width:100%; font-size:var(--font-size-xs); padding:6px 8px; margin-top:2px;
            background:var(--bg-tertiary); border:1px solid var(--border); border-radius:var(--radius-sm); color:var(--text-primary); font-family:inherit;
        }
        .cb-char-modal-body textarea { min-height:48px; resize:vertical; }
        .cb-char-row { display:flex; gap:6px; flex-wrap:wrap; margin-top:8px; align-items:center; }
        .cb-char-ref-thumb { max-width:64px; max-height:64px; border-radius:var(--radius-sm); border:1px solid var(--border); }
        .cb-modal-root {
            position:fixed; inset:0; z-index:500; display:flex; align-items:center; justify-content:center;
            padding:16px; box-sizing:border-box;
        }
        .cb-modal-root[hidden] { display:none !important; }
        .cb-modal-backdrop { position:absolute; inset:0; background:rgba(0,0,0,0.5); }
        .cb-modal-dialog {
            position:relative; z-index:1; width:min(480px,100%); max-height:min(88vh,760px); display:flex; flex-direction:column;
            background:var(--bg-secondary); border:1px solid var(--border); border-radius:var(--radius-lg);
            box-shadow:0 12px 40px rgba(0,0,0,0.28);
        }
        .cb-modal-header {
            display:flex; align-items:center; justify-content:space-between; padding:12px 16px;
            border-bottom:1px solid var(--border); flex-shrink:0; gap:12px;
        }
        .cb-modal-title { margin:0; font-size:15px; font-weight:600; color:var(--text-primary); }
        .cb-modal-close {
            border:none; background:transparent; font-size:22px; line-height:1; cursor:pointer;
            color:var(--text-secondary); padding:4px 10px; border-radius:var(--radius-sm);
        }
        .cb-modal-close:hover { color:var(--text-primary); background:var(--bg-tertiary); }
        .cb-modal-body { overflow-y:auto; padding:12px 16px 18px; flex:1; min-height:0; -webkit-overflow-scrolling:touch; }
        .cb-msg-image { padding:8px; }
        .cb-msg-image img { max-width:min(100%,420px); border-radius:var(--radius-md); border:1px solid var(--border); display:block; }
        .cb-msg-image.cb-msg-image-loading { color:var(--text-tertiary); font-size:var(--font-size-xs); }

        @media (max-width:1024px) {
            .cb-layout { flex-direction:column; gap:12px; min-height:0; }
            .cb-sidebar-left, .cb-sidebar-right { width:100%; max-height:320px; }
            .cb-sidebar-right { max-height:400px; }
            .cb-chat-stage { padding:0; order:-1; flex:1; min-height:min(60vh,520px); }
            .cb-chat { max-width:none; flex:1 1 auto; }
        }
        @media (max-width:768px) {
            .cb-layout { min-height:400px; }
            .cb-sidebar-left { max-height:260px; }
            .cb-sidebar-right .cb-sysprompt { display:none; }
            .cb-options { padding:10px 12px; }
        }
    `);
})();

