(function() {
    Toolbox.register({
        id: 'crypto',
        title: '암호화 / 복호화',
        category: 'feature',
        desc: '텍스트를 AES, Base64, URL 인코딩으로 암호화·복호화합니다',
        layout: 'form',
        icon: '<rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0110 0v4"/>',
        tabs: [
            {
                id: 'encrypt', label: '암호화',
                build(c) {
            Mdd.setMood('smug'); Mdd.say('암호화 시작이다냥... 히히');
                    Toolbox.select(c, {
                        id: 'encryptMethod', label: '방식',
                        options: [
                            { value: 'aes', label: 'AES-256 (비밀번호 기반)' },
                            { value: 'base64', label: 'Base64 (단순 인코딩)' },
                            { value: 'url', label: 'URL 인코딩 (웹용)' },
                        ],
                        onChange() { toggleEncryptFields(); }
                    });
                    Toolbox.field(c, { id: 'plainInput', label: '평문', placeholder: '암호화할 텍스트를 입력하세요' });

                    const passGroup = document.createElement('div');
                    passGroup.id = 'encryptAesFields';
                    Toolbox.field(passGroup, { tag: 'input', id: 'encryptPass', label: '비밀번호', placeholder: '비밀번호를 입력하세요', type: 'password' });

                    const trigger = document.createElement('button');
                    trigger.className = 'collapsible-trigger';
                    trigger.innerHTML = '<svg class="chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"/></svg>고급 설정';
                    trigger.onclick = function () { Toolbox.toggleCollapsible(this); };
                    passGroup.appendChild(trigger);

                    const body = document.createElement('div');
                    body.className = 'collapsible-body';
                    body.innerHTML = '<div class="field-group"><div class="field-row"><label class="field-label" style="margin-bottom:0">반복 횟수 (Iterations)</label><span class="range-value" id="iterVal">10,000</span></div><input type="range" id="iterSlider" min="1000" max="100000" step="1000" value="10000"></div>';
                    passGroup.appendChild(body);
                    c.appendChild(passGroup);

                    Toolbox.button(c, { text: '암호화', onclick: doEncrypt });
                    Toolbox.resultBox(c, 'encrypt');

                    requestAnimationFrame(() => {
                        const slider = document.getElementById('iterSlider');
                        if (slider) slider.oninput = function () { document.getElementById('iterVal').textContent = Number(this.value).toLocaleString(); };
                    });
                }
            },
            {
                id: 'decrypt', label: '복호화',
                build(c) {
                    Toolbox.select(c, {
                        id: 'decryptMethod', label: '방식',
                        options: [
                            { value: 'aes', label: 'AES-256 (비밀번호 기반)' },
                            { value: 'base64', label: 'Base64 (단순 디코딩)' },
                            { value: 'url', label: 'URL 디코딩 (웹용)' },
                        ],
                        onChange() { toggleDecryptFields(); }
                    });

                    const loadBtn = document.createElement('button');
                    loadBtn.className = 'btn btn-ghost'; loadBtn.textContent = 'DATA 불러오기';
                    loadBtn.onclick = loadFromTxt;
                    Toolbox.field(c, { id: 'cipherInput', label: '암호문', placeholder: '암호문 또는 Base64 문자열을 입력하세요', topRight: loadBtn, mono: true });

                    const passGroup = document.createElement('div');
                    passGroup.id = 'decryptAesFields';
                    Toolbox.field(passGroup, { tag: 'input', id: 'decryptPass', label: '비밀번호', placeholder: '비밀번호를 입력하세요', type: 'password' });
                    c.appendChild(passGroup);

                    Toolbox.button(c, { text: '복호화', onclick: doDecrypt, style: 'background:var(--success)' });
                    Toolbox.resultBox(c, 'decrypt');
                }
            }
        ]
    });

    window.toggleEncryptFields = function() {
        const method = document.getElementById('encryptMethod').value;
        document.getElementById('encryptAesFields').style.display = method === 'aes' ? '' : 'none';
    }

    window.toggleDecryptFields = function() {
        const method = document.getElementById('decryptMethod').value;
        document.getElementById('decryptAesFields').style.display = method === 'aes' ? '' : 'none';
    }

    window.doEncrypt = function() {
        const method = document.getElementById('encryptMethod').value;
        const text = document.getElementById('plainInput').value;
        if (!text) { Toolbox.showToast('텍스트를 입력해주세요.', 'error'); return; }

        if (method === 'base64') {
            try {
                const encoded = btoa(unescape(encodeURIComponent(text)));
                Toolbox.displayResult('encrypt', '인코딩 완료', encoded, null);
                Toolbox.showToast('인코딩 성공');
            } catch (e) {
                Toolbox.displayResult('encrypt', '오류', '인코딩 실패: ' + e.message, null, true);
                Toolbox.showToast('인코딩 실패', 'error');
            }
            return;
        }

        if (method === 'url') {
            try {
                const encoded = encodeURIComponent(text);
                Toolbox.displayResult('encrypt', '인코딩 완료', encoded, null);
                Toolbox.showToast('인코딩 성공');
            } catch (e) {
                Toolbox.displayResult('encrypt', '오류', '인코딩 실패: ' + e.message, null, true);
                Toolbox.showToast('인코딩 실패', 'error');
            }
            return;
        }

        const pass = document.getElementById('encryptPass').value;
        const iterations = parseInt(document.getElementById('iterSlider').value);
        if (!pass) { Toolbox.showToast('비밀번호를 입력해주세요.', 'error'); return; }

        const t0 = performance.now();
        try {
            const salt = CryptoJS.lib.WordArray.random(16);
            const iv = CryptoJS.lib.WordArray.random(16);
            const key = CryptoJS.PBKDF2(pass, salt, { keySize: 256 / 32, iterations, hasher: CryptoJS.algo.SHA256 });
            const encrypted = CryptoJS.AES.encrypt(text, key, { iv, mode: CryptoJS.mode.CBC, padding: CryptoJS.pad.Pkcs7 });

            const hex = salt.toString(CryptoJS.enc.Hex) + iv.toString(CryptoJS.enc.Hex) + iterations.toString(16).padStart(8, '0') + encrypted.ciphertext.toString(CryptoJS.enc.Hex);
            const result = CryptoJS.enc.Hex.parse(hex).toString(CryptoJS.enc.Base64);

            Toolbox.displayResult('encrypt', '암호화 완료', result, (performance.now() - t0) / 1000);
            Toolbox.showToast('암호화 성공');
        } catch (e) {
            Toolbox.displayResult('encrypt', '오류', '암호화 실패: ' + e.message, null, true);
            Toolbox.showToast('암호화 실패', 'error');
        }
    }

    window.doDecrypt = function() {
        const method = document.getElementById('decryptMethod').value;
        const input = document.getElementById('cipherInput').value;
        if (!input) { Toolbox.showToast('암호문을 입력해주세요.', 'error'); return; }

        if (method === 'base64') {
            try {
                const decoded = decodeURIComponent(escape(atob(input.trim())));
                Toolbox.displayResult('decrypt', '디코딩 완료', decoded, null);
                Toolbox.showToast('디코딩 성공');
            } catch (e) {
                Toolbox.displayResult('decrypt', '오류', '디코딩 실패: 올바른 Base64 문자열이 아닙니다.', null, true);
                Toolbox.showToast('디코딩 실패', 'error');
            }
            return;
        }

        if (method === 'url') {
            try {
                const decoded = decodeURIComponent(input.trim().replace(/\+/g, '%20'));
                Toolbox.displayResult('decrypt', '디코딩 완료', decoded, null);
                Toolbox.showToast('디코딩 성공');
            } catch (e) {
                Toolbox.displayResult('decrypt', '오류', '디코딩 실패: 올바르지 않은 형식입니다.', null, true);
                Toolbox.showToast('디코딩 실패', 'error');
            }
            return;
        }

        const pass = document.getElementById('decryptPass').value;
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

            Toolbox.displayResult('decrypt', `복호화 완료 · iterations: ${iterations.toLocaleString()}`, result, (performance.now() - t0) / 1000);
            Toolbox.showToast('복호화 성공');
        } catch (e) {
            Toolbox.displayResult('decrypt', '오류', '해독 실패: ' + e.message, null, true);
            Toolbox.showToast('복호화 실패', 'error');
        }
    }

    window.loadFromTxt = async function() {
        try {
            const res = await fetch('/assets/js/mathjax-config.json?t=' + Date.now());
            if (!res.ok) throw new Error('데이터 파일 로드에 실패했습니다.');
            document.getElementById('cipherInput').value = (await res.text()).trim();
            Toolbox.showToast('데이터 로드 완료');
        } catch (e) { Toolbox.showToast(e.message, 'error'); }
    }
})();
