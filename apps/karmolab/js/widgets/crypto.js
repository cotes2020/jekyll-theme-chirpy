(function() {
    async function loadFromTxt() {
        try {
            const res = await fetch('/assets/js/mathjax-config.json?t=' + Date.now());
            if (!res.ok) throw new Error('데이터 파일 로드에 실패했습니다.');
            document.getElementById('cryptoInput').value = (await res.text()).trim();
            Toolbox.showToast('데이터 로드 완료');
        } catch (e) { Toolbox.showToast(e.message, 'error'); }
    }
    window.loadFromTxt = loadFromTxt;

    function toggleCryptoFields() {
        const method = document.getElementById('cryptoMethod').value;
        const mode = document.getElementById('cryptoMode').value;
        const aesFields = document.getElementById('cryptoAesFields');
        const execBtn = document.getElementById('cryptoExecBtn');
        const inputEl = document.getElementById('cryptoInput');

        if (aesFields) aesFields.style.display = method === 'aes' ? '' : 'none';
        if (execBtn) execBtn.textContent = mode === 'encrypt' ? '암호화' : '복호화';
        if (inputEl) {
            inputEl.placeholder = mode === 'encrypt'
                ? '암호화할 평문을 입력하세요'
                : '암호문 또는 Base64 문자열을 입력하세요';
        }
    }
    window.toggleCryptoFields = toggleCryptoFields;

    function swapResultToInput() {
        const resultContent = document.getElementById('cryptoResultContent');
        const inputEl = document.getElementById('cryptoInput');
        const hiddenMode = document.getElementById('cryptoMode');
        const modeBtns = document.querySelectorAll('.crypto-mode-btn');
        if (!resultContent || !inputEl) return;
        const text = resultContent.textContent?.trim();
        if (!text) { Toolbox.showToast('복사할 결과가 없습니다.', 'error'); return; }
        inputEl.value = text;
        const nextMode = hiddenMode?.value === 'encrypt' ? 'decrypt' : 'encrypt';
        if (hiddenMode) hiddenMode.value = nextMode;
        modeBtns.forEach(b => { b.classList.toggle('active', b.dataset.mode === nextMode); });
        toggleCryptoFields();
        Toolbox.showToast('입력창으로 복사 & 모드 전환');
    }
    window.swapResultToInput = swapResultToInput;

    function doEncrypt(text, method) {
        if (method === 'base64') {
            try {
                const encoded = btoa(unescape(encodeURIComponent(text)));
                Toolbox.displayResult('crypto', '인코딩 완료', encoded, null);
                Toolbox.showToast('인코딩 성공');
            } catch (e) {
                Toolbox.displayResult('crypto', '오류', '인코딩 실패: ' + e.message, null, true);
                Toolbox.showToast('인코딩 실패', 'error');
            }
            return;
        }

        if (method === 'url') {
            try {
                const encoded = encodeURIComponent(text);
                Toolbox.displayResult('crypto', '인코딩 완료', encoded, null);
                Toolbox.showToast('인코딩 성공');
            } catch (e) {
                Toolbox.displayResult('crypto', '오류', '인코딩 실패: ' + e.message, null, true);
                Toolbox.showToast('인코딩 실패', 'error');
            }
            return;
        }

        const pass = document.getElementById('cryptoPass').value;
        const iterations = parseInt(document.getElementById('cryptoIterSlider')?.value || 10000);
        if (!pass) { Toolbox.showToast('비밀번호를 입력해주세요.', 'error'); return; }

        const t0 = performance.now();
        try {
            const salt = CryptoJS.lib.WordArray.random(16);
            const iv = CryptoJS.lib.WordArray.random(16);
            const key = CryptoJS.PBKDF2(pass, salt, { keySize: 256 / 32, iterations, hasher: CryptoJS.algo.SHA256 });
            const encrypted = CryptoJS.AES.encrypt(text, key, { iv, mode: CryptoJS.mode.CBC, padding: CryptoJS.pad.Pkcs7 });

            const hex = salt.toString(CryptoJS.enc.Hex) + iv.toString(CryptoJS.enc.Hex) + iterations.toString(16).padStart(8, '0') + encrypted.ciphertext.toString(CryptoJS.enc.Hex);
            const result = CryptoJS.enc.Hex.parse(hex).toString(CryptoJS.enc.Base64);

            Toolbox.displayResult('crypto', '암호화 완료', result, (performance.now() - t0) / 1000);
            Toolbox.showToast('암호화 성공');
        } catch (e) {
            Toolbox.displayResult('crypto', '오류', '암호화 실패: ' + e.message, null, true);
            Toolbox.showToast('암호화 실패', 'error');
        }
    }

    function doDecrypt(input, method) {
        if (method === 'base64') {
            try {
                const decoded = decodeURIComponent(escape(atob(input.trim())));
                Toolbox.displayResult('crypto', '디코딩 완료', decoded, null);
                Toolbox.showToast('디코딩 성공');
            } catch (e) {
                Toolbox.displayResult('crypto', '오류', '디코딩 실패: 올바른 Base64 문자열이 아닙니다.', null, true);
                Toolbox.showToast('디코딩 실패', 'error');
            }
            return;
        }

        if (method === 'url') {
            try {
                const decoded = decodeURIComponent(input.trim().replace(/\+/g, '%20'));
                Toolbox.displayResult('crypto', '디코딩 완료', decoded, null);
                Toolbox.showToast('디코딩 성공');
            } catch (e) {
                Toolbox.displayResult('crypto', '오류', '디코딩 실패: 올바르지 않은 형식입니다.', null, true);
                Toolbox.showToast('디코딩 실패', 'error');
            }
            return;
        }

        const pass = document.getElementById('cryptoPass').value;
        if (!pass) { Toolbox.showToast('비밀번호를 입력해주세요.', 'error'); return; }

        const t0 = performance.now();
        try {
            const hex = CryptoJS.enc.Base64.parse(input).toString(CryptoJS.enc.Hex);
            if (hex.length < 72) throw new Error('올바른 포맷의 암호문이 아닙니다.');

            const salt = CryptoJS.enc.Hex.parse(hex.substring(0, 32));
            const iv = CryptoJS.enc.Hex.parse(hex.substring(32, 64));
            const iterations = parseInt(hex.substring(64, 72), 16);
            const ciphertext = CryptoJS.enc.Hex.parse(hex.substring(72));

            if (isNaN(iterations) || iterations <= 0 || iterations > 1000000) throw new Error('잘못된 데이터입니다.');

            const key = CryptoJS.PBKDF2(pass, salt, { keySize: 256 / 32, iterations, hasher: CryptoJS.algo.SHA256 });
            const decrypted = CryptoJS.AES.decrypt(CryptoJS.lib.CipherParams.create({ ciphertext }), key, { iv, mode: CryptoJS.mode.CBC, padding: CryptoJS.pad.Pkcs7 });
            const result = decrypted.toString(CryptoJS.enc.Utf8);

            if (!result) throw new Error('비밀번호가 일치하지 않거나 데이터가 손상되었습니다.');

            Toolbox.displayResult('crypto', `복호화 완료 · iterations: ${iterations.toLocaleString()}`, result, (performance.now() - t0) / 1000);
            Toolbox.showToast('복호화 성공');
        } catch (e) {
            Toolbox.displayResult('crypto', '오류', '해독 실패: ' + e.message, null, true);
            Toolbox.showToast('복호화 실패', 'error');
        }
    }

    function doCrypto() {
        const mode = document.getElementById('cryptoMode').value;
        const method = document.getElementById('cryptoMethod').value;
        const text = document.getElementById('cryptoInput').value;

        if (!text) {
            Toolbox.showToast(mode === 'encrypt' ? '텍스트를 입력해주세요.' : '암호문을 입력해주세요.', 'error');
            return;
        }

        if (mode === 'encrypt') {
            doEncrypt(text, method);
        } else {
            doDecrypt(text, method);
        }
    }
    window.doCrypto = doCrypto;

    Toolbox.register({
        ...Toolbox.getLazyWidgetPublicMeta('crypto'),
        tabs: [
            {
                id: 'crypto',
                label: '암호화 / 복호화',
                build(c) {
                    Mdd.setMood('smug'); Mdd.say('암호화·복호화 시작이에요... 히히');

                    const modeGroup = document.createElement('div');
                    modeGroup.className = 'field-group';
                    modeGroup.innerHTML = '<label class="field-label">모드</label>';
                    const modeWrap = document.createElement('div');
                    modeWrap.className = 'crypto-mode-btns';
                    modeWrap.style.display = 'flex'; modeWrap.style.gap = '8px';
                    const encBtn = document.createElement('button');
                    encBtn.type = 'button'; encBtn.className = 'btn crypto-mode-btn active';
                    encBtn.textContent = '암호화'; encBtn.dataset.mode = 'encrypt';
                    const decBtn = document.createElement('button');
                    decBtn.type = 'button'; decBtn.className = 'btn crypto-mode-btn';
                    decBtn.textContent = '복호화'; decBtn.dataset.mode = 'decrypt';
                    const hiddenMode = document.createElement('input');
                    hiddenMode.type = 'hidden'; hiddenMode.id = 'cryptoMode'; hiddenMode.value = 'encrypt';
                    [encBtn, decBtn].forEach(btn => {
                        btn.onclick = function() {
                            modeWrap.querySelectorAll('.crypto-mode-btn').forEach(b => b.classList.remove('active'));
                            btn.classList.add('active');
                            hiddenMode.value = btn.dataset.mode;
                            toggleCryptoFields();
                        };
                        modeWrap.appendChild(btn);
                    });
                    modeGroup.appendChild(modeWrap);
                    modeGroup.appendChild(hiddenMode);
                    c.appendChild(modeGroup);

                    Mdd.injectCSS('crypto-mode', '.crypto-mode-btn { background:var(--bg-tertiary); border:1px solid var(--border); color:var(--text-secondary); }.crypto-mode-btn:hover { background:var(--bg-hover); color:var(--text-primary); }.crypto-mode-btn.active { background:var(--accent); color:#fff; border-color:var(--accent); }');

                    const methodGroup = document.createElement('div');
                    methodGroup.className = 'field-group';
                    methodGroup.innerHTML = '<label class="field-label">방식</label>';
                    const methodWrap = document.createElement('div');
                    methodWrap.className = 'crypto-method-btns';
                    methodWrap.style.display = 'flex'; methodWrap.style.gap = '8px'; methodWrap.style.flexWrap = 'wrap';
                    const methods = [
                        { value: 'base64', label: 'Base64' },
                        { value: 'aes', label: 'AES-256' },
                        { value: 'url', label: 'URL' },
                    ];
                    const hiddenMethod = document.createElement('input');
                    hiddenMethod.type = 'hidden'; hiddenMethod.id = 'cryptoMethod'; hiddenMethod.value = 'base64';
                    methods.forEach((m, i) => {
                        const btn = document.createElement('button');
                        btn.type = 'button';
                        btn.className = 'btn crypto-mode-btn crypto-method-btn' + (i === 0 ? ' active' : '');
                        btn.textContent = m.label;
                        btn.dataset.method = m.value;
                        btn.onclick = function() {
                            methodWrap.querySelectorAll('.crypto-method-btn').forEach(b => b.classList.remove('active'));
                            btn.classList.add('active');
                            hiddenMethod.value = btn.dataset.method;
                            toggleCryptoFields();
                        };
                        methodWrap.appendChild(btn);
                    });
                    methodGroup.appendChild(methodWrap);
                    methodGroup.appendChild(hiddenMethod);
                    c.appendChild(methodGroup);

                    const loadBtn = document.createElement('button');
                    loadBtn.className = 'btn btn-ghost'; loadBtn.textContent = 'DATA 불러오기';
                    loadBtn.onclick = function () { window.loadFromTxt(); };

                    Toolbox.field(c, {
                        id: 'cryptoInput', label: '입력',
                        placeholder: '암호화할 평문 또는 복호화할 암호문을 입력하세요',
                        topRight: loadBtn, mono: true
                    });

                    const passGroup = document.createElement('div');
                    passGroup.id = 'cryptoAesFields';
                    Toolbox.field(passGroup, { tag: 'input', id: 'cryptoPass', label: '비밀번호', placeholder: '비밀번호를 입력하세요', type: 'password' });

                    const trigger = document.createElement('button');
                    trigger.className = 'collapsible-trigger';
                    trigger.innerHTML = '<svg class="chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"/></svg>고급 설정';
                    trigger.onclick = function () { Toolbox.toggleCollapsible(this); };
                    passGroup.appendChild(trigger);

                    const body = document.createElement('div');
                    body.className = 'collapsible-body';
                    body.innerHTML = '<div class="field-group"><div class="field-row"><label class="field-label" style="margin-bottom:0">반복 횟수 (Iterations)</label><span class="range-value" id="cryptoIterVal">10,000</span></div><input type="range" id="cryptoIterSlider" min="1000" max="100000" step="1000" value="10000"></div>';
                    passGroup.appendChild(body);
                    c.appendChild(passGroup);

                    const btnRow = document.createElement('div');
                    btnRow.className = 'field-group';
                    btnRow.style.display = 'flex'; btnRow.style.gap = '8px'; btnRow.style.flexWrap = 'wrap';
                    const execBtn = document.createElement('button');
                    execBtn.className = 'btn btn-primary';
                    execBtn.id = 'cryptoExecBtn';
                    execBtn.textContent = '실행';
                    execBtn.onclick = function () { window.doCrypto(); };
                    btnRow.appendChild(execBtn);
                    const swapBtn = document.createElement('button');
                    swapBtn.className = 'btn btn-ghost';
                    swapBtn.textContent = '결과를 입력으로';
                    swapBtn.onclick = function () { window.swapResultToInput(); };
                    btnRow.appendChild(swapBtn);
                    c.appendChild(btnRow);

                    Toolbox.resultBox(c, 'crypto');

                    requestAnimationFrame(() => {
                        window.toggleCryptoFields();
                        const slider = document.getElementById('cryptoIterSlider');
                        if (slider) slider.oninput = function () { document.getElementById('cryptoIterVal').textContent = Number(this.value).toLocaleString(); };
                    });
                }
            }
        ]
    });

    window.toggleCryptoFields = toggleCryptoFields;
    window.swapResultToInput = swapResultToInput;
    window.doCrypto = doCrypto;
    window.loadFromTxt = loadFromTxt;
})();
