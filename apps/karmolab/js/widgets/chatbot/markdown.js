/** 챗봇 메시지용 경량 마크다운 → HTML (styles.js의 .cb-msg-bot 규칙과 대응) */
(function () {
    window.ChatbotMarkdown = window.ChatbotMarkdown || {};

    function escapeHtml(s) {
        if (typeof Toolbox !== 'undefined' && Toolbox.escapeHtml) return Toolbox.escapeHtml(s);
        return String(s)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function renderMarkdown(text) {
        const codeBlocks = [];
        let md = text.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
            const cls = lang ? ` language-${lang}` : '';
            const langLabel = lang || 'code';
            const header = `<div class="cb-code-header"><span class="cb-code-lang">${escapeHtml(langLabel)}</span><button class="btn btn-ghost" onclick="navigator.clipboard.writeText(this.closest('.cb-code-block').querySelector('code').textContent).then(()=>Toolbox.showToast('복사됨'))">복사</button></div>`;
            codeBlocks.push(`<div class="cb-code-block">${header}<pre class="${cls.trim()}"><code class="${cls.trim()}">${escapeHtml(code.trimEnd())}</code></pre></div>`);
            return `%%CODEBLOCK_${codeBlocks.length - 1}%%`;
        });

        const inlineCodes = [];
        md = md.replace(/`([^`]+)`/g, (_, code) => {
            inlineCodes.push(`<code>${escapeHtml(code)}</code>`);
            return `%%INLINE_${inlineCodes.length - 1}%%`;
        });

        md = escapeHtml(md);

        md = md.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        md = md.replace(/\*(.+?)\*/g, '<em>$1</em>');

        md = md.replace(/\[(.+?)\]\((.+?)\)/g, (_, t, url) => {
            if (/^(https?:|mailto:|\/|#)/i.test(url)) return `<a href="${url}" target="_blank" rel="noopener">${t}</a>`;
            return `${t} (${url})`;
        });

        md = md.replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>');
        md = md.replace(/^---$/gm, '<hr>');

        md = md.replace(/(^\|.+\|$\n?)+/gm, tableBlock => {
            const rows = tableBlock.trim().split('\n').filter(r => r.trim());
            if (rows.length < 2) return tableBlock;
            const isSep = r => /^\|[\s\-:|]+\|$/.test(r.trim());
            const parseRow = (r, tag) => {
                const cells = r.trim().replace(/^\||\|$/g, '').split('|').map(c => c.trim());
                return '<tr>' + cells.map(c => `<${tag}>${c}</${tag}>`).join('') + '</tr>';
            };
            let html = '<table>';
            let headerDone = false;
            for (let i = 0; i < rows.length; i++) {
                if (isSep(rows[i])) { headerDone = true; continue; }
                if (!headerDone && i === 0 && rows.length > 1 && isSep(rows[1])) {
                    html += '<thead>' + parseRow(rows[i], 'th') + '</thead><tbody>';
                    headerDone = true;
                    i++;
                    continue;
                }
                html += parseRow(rows[i], 'td');
            }
            html += '</tbody></table>';
            return html;
        });

        md = md.replace(/^[\-\*] (.+)$/gm, '<li>$1</li>');
        md = md.replace(/(<li>.*<\/li>\n?)+/g, m => `<ul>${m}</ul>`);
        md = md.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');

        md = md.replace(/^### (.+)$/gm, '<strong>$1</strong>');
        md = md.replace(/^## (.+)$/gm, '<strong>$1</strong>');
        md = md.replace(/^# (.+)$/gm, '<strong style="font-size:15px">$1</strong>');

        md = md.replace(/\n\n+/g, '</p><p>');
        md = md.replace(/\n/g, '<br>');
        md = `<p>${md}</p>`;

        codeBlocks.forEach((block, i) => { md = md.replace(`%%CODEBLOCK_${i}%%`, block); });
        inlineCodes.forEach((code, i) => { md = md.replace(`%%INLINE_${i}%%`, code); });

        md = md.replace(/<p>\s*<\/p>/g, '');

        return md;
    }

    window.ChatbotMarkdown.renderMarkdown = renderMarkdown;
})();
