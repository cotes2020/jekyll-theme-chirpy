// @ts-nocheck
(function () {
    const T = window.Tierlist = window.Tierlist || {};

    let ctxMenu = null;

    function hideContextMenu() {
        if (ctxMenu) { ctxMenu.remove(); ctxMenu = null; }
    }

    function showContextMenu(x, y, actions) {
        hideContextMenu();
        ctxMenu = document.createElement('div');
        ctxMenu.className = 'tl-ctx';
        ctxMenu.addEventListener('pointerdown', (ev) => ev.stopPropagation());

        actions.forEach(a => {
            if (a === 'sep') { const s = document.createElement('div'); s.className = 'tl-ctx-sep'; ctxMenu.appendChild(s); return; }
            const btn = document.createElement('button');
            btn.className = 'tl-ctx-item' + (a.danger ? ' danger' : '');
            btn.textContent = a.label;
            btn.onpointerdown = (ev) => {
                ev.preventDefault();
                ev.stopPropagation();
                hideContextMenu();
                a.action();
            };
            ctxMenu.appendChild(btn);
        });

        document.body.appendChild(ctxMenu);
        const rect = ctxMenu.getBoundingClientRect();
        if (x + rect.width > window.innerWidth) x = window.innerWidth - rect.width - 8;
        if (y + rect.height > window.innerHeight) y = window.innerHeight - rect.height - 8;
        ctxMenu.style.left = x + 'px';
        ctxMenu.style.top = y + 'px';

        setTimeout(() => document.addEventListener('pointerdown', hideContextMenu, { once: true }), 0);
    }

    function openDialog({ title, bodyHtml, wide, onMount }) {
        const overlay = document.createElement('div');
        overlay.className = 'tl-dialog-overlay';
        const dialog = document.createElement('div');
        dialog.className = 'tl-dialog' + (wide ? ' tl-dialog-wide' : '');
        dialog.innerHTML = `<h3>${Toolbox.escapeHtml(title)}</h3>${bodyHtml || ''}`;
        overlay.appendChild(dialog);
        document.body.appendChild(overlay);
        overlay.onclick = e => { if (e.target === overlay) overlay.remove(); };
        const api = { overlay, dialog, close: () => overlay.remove() };
        onMount?.(api);
        return api;
    }

    T.ui = { showContextMenu, hideContextMenu, openDialog };
})();

