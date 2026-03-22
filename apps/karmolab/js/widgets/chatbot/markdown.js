/** Chatbot 경량 마크다운 파서 */
window.Chatbot = window.Chatbot || {};
window.Chatbot.Markdown = (function () {
    var escapeHtml = function (s) {
        return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
    };

    function render(text) {
        var codeBlocks = [];
        var md = text.replace(/```(\w*)\n([\s\S]*?)```/g, function (_, lang, code) {
            var cls = lang ? (' language-' + lang) : '';
            var langLabel = lang || 'code';
            var header = '<div class="cb-code-header"><span class="cb-code-lang">' + escapeHtml(langLabel) + '</span><button class="btn btn-ghost" onclick="navigator.clipboard.writeText(this.closest(\'.cb-code-block\').querySelector(\'code\').textContent).then(function(){Toolbox.showToast(\'복사됨\')})">복사</button></div>';
            codeBlocks.push('<div class="cb-code-block">' + header + '<pre class="' + cls.trim() + '"><code class="' + cls.trim() + '">' + escapeHtml(code.trimEnd()) + '</code></pre></div>');
            return '%%CODEBLOCK_' + (codeBlocks.length - 1) + '%%';
        });
        var inlineCodes = [];
        md = md.replace(/`([^`]+)`/g, function (_, code) {
            inlineCodes.push('<code>' + escapeHtml(code) + '</code>');
            return '%%INLINE_' + (inlineCodes.length - 1) + '%%';
        });
        md = escapeHtml(md);
        md = md.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>').replace(/\*(.+?)\*/g, '<em>$1</em>');
        md = md.replace(/\[(.+?)\]\((.+?)\)/g, function (_, text, url) {
            if (/^(https?:|mailto:|\/|#)/i.test(url)) return '<a href="' + url + '" target="_blank" rel="noopener">' + text + '</a>';
            return text + ' (' + url + ')';
        });
        md = md.replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>');
        md = md.replace(/^---$/gm, '<hr>');
        md = md.replace(/(^\|.+\|$\n?)+/gm, function (tableBlock) {
            var rows = tableBlock.trim().split('\n').filter(function (r) { return r.trim(); });
            if (rows.length < 2) return tableBlock;
            var isSep = function (r) { return /^\|[\s\-:|]+\|$/.test(r.trim()); };
            var parseRow = function (r, tag) {
                var cells = r.trim().replace(/^\||\|$/g, '').split('|').map(function (c) { return c.trim(); });
                return '<tr>' + cells.map(function (c) { return '<' + tag + '>' + c + '</' + tag + '>'; }).join('') + '</tr>';
            };
            var html = '<table>';
            var headerDone = false;
            for (var i = 0; i < rows.length; i++) {
                if (isSep(rows[i])) { headerDone = true; continue; }
                if (!headerDone && i === 0 && rows.length > 1 && isSep(rows[1])) {
                    html += '<thead>' + parseRow(rows[i], 'th') + '</thead><tbody>';
                    headerDone = true; i++;
                    continue;
                }
                html += parseRow(rows[i], 'td');
            }
            html += '</tbody></table>';
            return html;
        });
        md = md.replace(/^[\-\*] (.+)$/gm, '<li>$1</li>');
        md = md.replace(/(<li>.*<\/li>\n?)+/g, function (m) { return '<ul>' + m + '</ul>'; });
        md = md.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');
        md = md.replace(/^### (.+)$/gm, '<strong>$1</strong>').replace(/^## (.+)$/gm, '<strong>$1</strong>').replace(/^# (.+)$/gm, '<strong style="font-size:15px">$1</strong>');
        md = md.replace(/\n\n+/g, '</p><p>').replace(/\n/g, '<br>');
        md = '<p>' + md + '</p>';
        codeBlocks.forEach(function (block, i) { md = md.replace('%%CODEBLOCK_' + i + '%%', block); });
        inlineCodes.forEach(function (code, i) { md = md.replace('%%INLINE_' + i + '%%', code); });
        md = md.replace(/<p>\s*<\/p>/g, '');
        return md;
    }

    return { render: render, escapeHtml: escapeHtml };
})();
