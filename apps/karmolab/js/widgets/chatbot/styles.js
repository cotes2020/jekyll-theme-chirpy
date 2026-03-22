/** Chatbot 위젯 스타일 */
(function () {
    Mdd.injectCSS('chatbot', `
        .cb-layout { display:flex; gap:20px; flex:1; min-height:0; }
        .cb-sidebar { width:260px; flex-shrink:0; display:flex; flex-direction:column; border:1px solid var(--border); background:var(--bg-secondary); overflow:hidden; }
        .cb-sidebar-header { padding:16px; border-bottom:1px solid var(--border); }
        .cb-sidebar-header h3 { font-size:var(--font-size-sm); font-weight:600; color:var(--text-primary); margin-bottom:12px; }
        .cb-model-label { font-size:var(--font-size-xs); font-weight:500; color:var(--text-secondary); margin:12px 0 4px; display:block; }
        .cb-options { padding:12px 16px; border-bottom:1px solid var(--border); }
        .cb-option-row { display:flex; align-items:center; justify-content:space-between; margin-bottom:8px; }
        .cb-option-row label { font-size:var(--font-size-xs); color:var(--text-secondary); }
        .cb-info-btn { width:18px; height:18px; border-radius:50%; border:1px solid var(--border); background:var(--bg-tertiary); color:var(--text-tertiary); font-size:11px; font-weight:600; cursor:pointer; display:inline-flex; align-items:center; justify-content:center; margin-left:4px; line-height:1; padding:0; font-family:inherit; transition:all 0.2s; }
        .cb-info-btn:hover { border-color:var(--accent); color:var(--accent); }
        .cb-info-card { display:none; margin-top:8px; padding:10px 12px; font-size:var(--font-size-xs); line-height:1.5; background:var(--bg-primary); border:1px solid var(--border); border-radius:var(--radius-md); color:var(--text-secondary); }
        .cb-info-card.open { display:block; }
        .cb-info-card strong { color:var(--text-primary); }
        .cb-info-card table { width:100%; margin:8px 0; font-size:var(--font-size-2xs); }
        .cb-info-card td { padding:2px 8px 2px 0; vertical-align:top; }
        .cb-toggle { position:relative; width:36px; height:20px; }
        .cb-toggle input { opacity:0; width:0; height:0; }
        .cb-toggle-slider { position:absolute; inset:0; background:var(--bg-active); border-radius:10px; cursor:pointer; transition:background 0.2s; }
        .cb-toggle-slider::before { content:''; position:absolute; width:16px; height:16px; border-radius:50%; background:#fff; left:2px; top:2px; transition:transform 0.2s; }
        .cb-toggle input:checked + .cb-toggle-slider { background:var(--accent); }
        .cb-toggle input:checked + .cb-toggle-slider::before { transform:translateX(16px); }
        .cb-sysprompt { flex:1; padding:12px 16px; overflow-y:auto; display:flex; flex-direction:column; }
        .cb-sysprompt label { font-size:var(--font-size-xs); font-weight:500; color:var(--text-secondary); margin-bottom:4px; display:block; }
        .cb-sysprompt textarea { flex:1; min-height:120px; font-size:var(--font-size-xs); line-height:1.5; resize:none; padding:8px; background:var(--bg-tertiary); }
        .cb-chat { flex:1; display:flex; flex-direction:column; background:var(--bg-primary); border:1px solid var(--border); overflow:hidden; }
        .cb-chat-header { padding:12px 16px; border-bottom:1px solid var(--border); display:flex; align-items:center; justify-content:space-between; background:var(--bg-secondary); }
        .cb-chat-header-title { font-size:var(--font-size-sm); font-weight:600; color:var(--text-primary); }
        .cb-messages { flex:1; overflow-y:auto; padding:16px; display:flex; flex-direction:column; gap:10px; }
        .cb-msg { max-width:80%; padding:10px 14px; border-radius:14px; font-size:var(--font-size-sm); line-height:1.6; animation:fadeIn 0.3s ease; word-break:break-word; }
        .cb-msg-user { align-self:flex-end; background:var(--accent); color:#fff; border-bottom-right-radius:4px; white-space:pre-wrap; }
        .cb-msg-bot { align-self:flex-start; background:var(--bg-tertiary); color:var(--text-primary); border:1px solid var(--border); border-bottom-left-radius:4px; }
        .cb-msg-error { background:var(--error-subtle) !important; border-color:var(--error) !important; color:var(--error) !important; }
        .cb-msg-bot p { margin:0 0 8px; } .cb-msg-bot p:last-child { margin:0; }
        .cb-msg-bot strong { font-weight:700; } .cb-msg-bot em { font-style:italic; }
        .cb-msg-bot code { background:rgba(255,255,255,0.08); padding:1px 5px; border-radius:3px; font-family:'Cascadia Code','Consolas',monospace; font-size:var(--font-size-xs); }
        .cb-msg-bot pre { border:1px solid var(--border); border-radius:var(--radius-sm); padding:0; margin:8px 0; overflow-x:auto; position:relative; }
        .cb-msg-bot pre code { background:none !important; padding:10px 12px; font-size:var(--font-size-xs); line-height:1.5; display:block; }
        .cb-code-header { display:flex; justify-content:space-between; align-items:center; padding:4px 8px 4px 12px; background:rgba(255,255,255,0.04); border-bottom:1px solid var(--border); font-size:var(--font-size-2xs); }
        .cb-code-lang { color:var(--text-tertiary); font-weight:600; text-transform:uppercase; letter-spacing:0.5px; }
        .cb-msg-bot ul, .cb-msg-bot ol { margin:6px 0; padding-left:20px; } .cb-msg-bot li { margin:2px 0; }
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
        .cb-typing span { display:inline-block; width:5px; height:5px; background:var(--text-tertiary); border-radius:50%; animation:cbTyping 1.4s infinite ease-in-out both; margin:0 1px; }
        .cb-typing span:nth-child(1) { animation-delay:-0.32s; } .cb-typing span:nth-child(2) { animation-delay:-0.16s; }
        @keyframes cbTyping { 0%,80%,100%{transform:scale(0)} 40%{transform:scale(1)} }
        .cb-input-area { padding:12px 16px; border-top:1px solid var(--border); background:var(--bg-secondary); }
        .cb-input-row { display:flex; gap:8px; }
        .cb-input-row textarea { flex:1; min-height:40px; max-height:80px; resize:none; font-size:var(--font-size-sm); padding:8px 12px; }
        .cb-send-btn { width:40px; background:var(--accent); color:#fff; border:none; border-radius:var(--radius-md); cursor:pointer; font-size:16px; display:flex; align-items:center; justify-content:center; transition:background var(--transition); }
        .cb-send-btn:hover { background:var(--accent-hover); }
        .cb-token-bar { display:flex; justify-content:space-between; margin-top:6px; font-size:var(--font-size-2xs); color:var(--text-tertiary); font-family:monospace; }
        .cb-mic-btn { width:40px; background:transparent; color:var(--text-secondary); border:1px solid var(--border); border-radius:var(--radius-md); cursor:pointer; font-size:16px; display:flex; align-items:center; justify-content:center; transition:all var(--transition); }
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
        .cb-search-bar { display:none; padding:8px 16px; border-bottom:1px solid var(--border); background:var(--bg-secondary); }
        .cb-search-bar.open { display:flex; gap:6px; align-items:center; }
        .cb-search-bar input { flex:1; font-size:var(--font-size-xs); padding:6px 10px; border:1px solid var(--border); border-radius:var(--radius-sm); background:var(--bg-primary); color:var(--text-primary); outline:none; font-family:inherit; }
        .cb-search-bar input:focus { border-color:var(--accent); }
        .cb-search-bar .cb-search-nav { font-size:var(--font-size-xs); color:var(--text-tertiary); white-space:nowrap; }
        .cb-search-highlight { background:rgba(255,200,0,0.35); border-radius:2px; padding:0 1px; }
        .cb-search-highlight.current { background:rgba(255,200,0,0.7); outline:2px solid var(--accent); }
        .cb-shortcuts-overlay { display:none; position:absolute; top:0; left:0; right:0; bottom:0; background:rgba(0,0,0,0.6); z-index:100; align-items:center; justify-content:center; }
        .cb-shortcuts-overlay.open { display:flex; }
        .cb-shortcuts-panel { background:var(--bg-secondary); border:1px solid var(--border); border-radius:var(--radius-lg); padding:20px 24px; max-width:340px; width:90%; }
        .cb-shortcuts-panel h3 { margin:0 0 12px; font-size:14px; }
        .cb-shortcut-row { display:flex; justify-content:space-between; align-items:center; padding:4px 0; font-size:var(--font-size-xs); }
        .cb-shortcut-key { display:inline-block; padding:2px 8px; border-radius:4px; font-size:var(--font-size-xs); font-family:monospace; background:var(--bg-primary); border:1px solid var(--border); color:var(--text-secondary); min-width:24px; text-align:center; }
        .cb-msg-wrap { position:relative; }
        .cb-msg-copy { position:absolute; top:6px; right:6px; opacity:0; transition:opacity 0.2s; }
        .cb-msg-wrap:hover .cb-msg-copy { opacity:1; }
        @media (max-width:768px) { .cb-layout { flex-direction:column; min-height:400px; } .cb-sidebar { width:100%; max-height:200px; flex-direction:row; } .cb-sidebar-header { flex:1; } .cb-sysprompt { display:none; } .cb-options { border-bottom:none; border-right:1px solid var(--border); padding:8px; } }
    `);
})();
