document.addEventListener('DOMContentLoaded', () => {
    // 1. DOM 요소 가져오기
    const problemElement = document.getElementById('problem');
    const userInputDisplayElement = document.getElementById('user-input-display');
    const timerElement = document.getElementById('timer');
    const cpmElement = document.getElementById('cpm');
    const restartBtn = document.getElementById('restart-btn');
    const themeToggleBtn = document.getElementById('theme-toggle-btn');
    const virtualKeyboardElement = document.getElementById('virtual-keyboard');
    const body = document.body;

    // 2. 월배열 2-263 키 매핑 및 연습 문장 (외부 파일에서 로드)
    let tsukiLayout = {};
    let sentences = [];

    // 3. 상태 변수
    let currentNormalSentence = '';
    let currentHiraganaSentence = '';
    let currentCntTypedFont = [];
    let userInput = '';
    let timer;
    let timeElapsed = 0;
    let totalTypedChars = 0; // 총 입력된 문자 수
    let totalCorrectChars = 0; // 총 정확하게 입력된 문자 수
    let gameStarted = false;
    let MiddleShiftActive = false;
    let keyPressed = {};

    const dakutenMap = {
        'か': 'が', 'き': 'ぎ', 'く': 'ぐ', 'け': 'げ', 'こ': 'ご',
        'さ': 'ざ', 'し': 'じ', 'す': 'ず', 'せ': 'ぜ', 'そ': 'ぞ',
        'た': 'だ', 'ち': 'ぢ', 'つ': 'づ', 'て': 'で', 'と': 'ど',
        'は': 'ば', 'ひ': 'び', 'ふ': 'ぶ', 'へ': 'べ', 'ほ': 'ぼ'
    };

    const handakutenMap = {
        'は': 'ぱ', 'ひ': 'ぴ', 'ふ': 'ぷ', 'へ': 'ぺ', 'ほ': 'ぽ'
    };
    let reverseDakutenMap = {};
    let reverseHandakutenMap = {};

    // Key Codes for special actions
    const KEY_CODE_MIDDLE_SHIFT_LEFT = 'KeyD';
    const KEY_CODE_MIDDLE_SHIFT_RIGHT = 'KeyK';
    const KEY_CODE_DAKUTEN = 'KeyL';
    const KEY_CODE_HANDAKUTEN = 'Slash';
    const KEY_CODE_BACKSPACE = 'Backspace';
    const KEY_CODE_ENTER = 'Enter';
    const KEY_CODE_SPACE = 'Space';

    // 4. 핵심 로직 (함수 정의)

    // 현재까지 정확히 입력된 길이를 반환하는 헬퍼 함수
    function getCorrectLength() {
        let len = 0;
        while (len < userInput.length &&
               len < currentHiraganaSentence.length &&
               userInput[len] === currentHiraganaSentence[len]) {
            len++;
        }
        return len;
    }

    function findKeyForChar(char) {
        for (const keyCode in tsukiLayout) {
            const key = tsukiLayout[keyCode];
            if (key.normal === char) {
                return { keyCode, shift: false };
            }
            if (key.shift === char) {
                return { keyCode, shift: true };
            }
        }
        return null;
    }

    function getKeysForChar(char) {
        const keys = [];
        const baseCharDakuten = reverseDakutenMap[char];
        const baseCharHandakuten = reverseHandakutenMap[char];
        const baseChar = baseCharDakuten || baseCharHandakuten || char;

        const keyInfo = findKeyForChar(baseChar);
        if (!keyInfo) return [];

        if (keyInfo.shift) {
            keys.push({ keyCode: 'MiddleShift' }); // 1. Generic Shift Key
        }
        keys.push({ keyCode: keyInfo.keyCode }); // 2. Character Key

        if (baseCharDakuten) {
            keys.push({ keyCode: 'KeyL' }); // 3. Dakuten Key
        } else if (baseCharHandakuten) {
            keys.push({ keyCode: 'Slash' }); // 3. Handakuten Key
        }

        return keys;
    }

    function updateKeyboardGuide() {
        // 1. Clear all existing guide classes
        document.querySelectorAll('.keyboard-key').forEach(key => {
            key.classList.remove('guide', 'guide-next');
        });

        const correctLength = getCorrectLength();
        if (correctLength >= currentHiraganaSentence.length) return; // No more characters to type

        const nextChar = currentHiraganaSentence[correctLength];
        const requiredKeys = getKeysForChar(nextChar);

        if (requiredKeys.length === 0) return; // No keys found for the character

        // Determine the current step in the key sequence for the next character
        let currentStep = 0;
        if (MiddleShiftActive) {
            currentStep = 1; // If MiddleShift is active, we are at the second step of the sequence
        } else {
            // Check if the base character for dakuten/handakuten has already been typed
            const baseChar = reverseDakutenMap[nextChar] || reverseHandakutenMap[nextChar];
            if (baseChar && userInput[correctLength] === baseChar) {
                currentStep = requiredKeys.length - 1; // If base char is typed, next is dakuten/handakuten
            }
        }

        // The key at currentStep is the main guide
        const mainGuideKeyCode = requiredKeys[currentStep].keyCode;

        // All keys after currentStep are auxiliary guides
        const auxiliaryGuideKeyCodes = [];
        for (let i = currentStep + 1; i < requiredKeys.length; i++) {
            auxiliaryGuideKeyCodes.push(requiredKeys[i].keyCode);
        }

        // Determine the actual key codes that should receive the 'guide' class
        let actualMainGuideKeyCodes = [];
        if (mainGuideKeyCode === 'MiddleShift') {
            actualMainGuideKeyCodes.push(KEY_CODE_MIDDLE_SHIFT_LEFT);
            actualMainGuideKeyCodes.push(KEY_CODE_MIDDLE_SHIFT_RIGHT);
        } else {
            actualMainGuideKeyCodes.push(mainGuideKeyCode);
        }

        // Filter auxiliaryGuideKeyCodes to ensure no overlap with actualMainGuideKeyCodes
        const filteredAuxiliaryGuideKeyCodes = auxiliaryGuideKeyCodes.filter(keyCode =>
            !actualMainGuideKeyCodes.includes(keyCode)
        );

        // Apply auxiliary guides first (guide-next)
        filteredAuxiliaryGuideKeyCodes.forEach(keyCode => {
            const keyElement = document.getElementById(`key-${keyCode}`);
            if (keyElement) {
                keyElement.classList.add('guide-next');
            }
        });

        // Apply main guides second (guide)
        actualMainGuideKeyCodes.forEach(keyCode => {
            const keyElement = document.getElementById(`key-${keyCode}`);
            if (keyElement) {
                keyElement.classList.add('guide');
            }
        });
    }


    // 데이터 로드 및 초기화
    async function initializeApp() {
        try {
            const [keymapResponse, sentencesResponse] = await Promise.all([
                fetch('keymap.json'),
                fetch('practice-text.json')
            ]);

            tsukiLayout = await keymapResponse.json();
            const sentencesData = await sentencesResponse.json();
            sentences = sentencesData.sentences;

            // 역방향 맵 생성
            for (const key in dakutenMap) {
                reverseDakutenMap[dakutenMap[key]] = key;
            }
            for (const key in handakutenMap) {
                reverseHandakutenMap[handakutenMap[key]] = key;
            }


            createVirtualKeyboard(); // 키맵 로드 후 가상 키보드 생성
            startNewGame(); // 모든 데이터 로드 후 새 게임 시작
        } catch (error) {
            console.error('ロードに失敗しました。', error);
            problemElement.textContent = 'ロードに失敗しました。';
        }
    }

    // 새 게임 시작
    function startNewGame() {
        userInput = '';
        MiddleShiftActive = false; // 시작 시 시프트 초기화
        keyPressed = {}; // 시작 시 눌린 키 상태 초기화
        
        userInputDisplayElement.textContent = '';

        const randomIndex = Math.floor(Math.random() * sentences.length);
        const selectedSentence = sentences[randomIndex];
        currentNormalSentence = selectedSentence.normal;
        currentHiraganaSentence = selectedSentence.hiragana;
        currentCntTypedFont = selectedSentence.cnttypedfont;
        
        problemElement.innerHTML = '';

        const normalTextDiv = document.createElement('div');
        normalTextDiv.className = 'normal-text';
        currentNormalSentence.split('').forEach(char => {
            const charSpan = document.createElement('span');
            charSpan.textContent = char;
            normalTextDiv.appendChild(charSpan);
        });
        problemElement.appendChild(normalTextDiv);

        const hiraganaTextDiv = document.createElement('div');
        hiraganaTextDiv.className = 'hiragana-text';
        currentHiraganaSentence.split('').forEach(char => {
            const charSpan = document.createElement('span');
            charSpan.textContent = char;
            hiraganaTextDiv.appendChild(charSpan);
        });
        problemElement.appendChild(hiraganaTextDiv);

        if (normalTextDiv.children.length > 0) {
            normalTextDiv.children[0].classList.add('current');
        }

        if (hiraganaTextDiv.children.length > 0) {
            hiraganaTextDiv.children[0].classList.add('current');
        }
        updateKeyboardGuide();
    }

    // 입력 처리
    function handleKeyDown(e) {
        e.preventDefault(); // 모든 키 입력에 대한 기본 동작을 여기서 한 번에 막습니다.

        // 가상 키보드 활성화 효과
        const keyElement = document.getElementById(`key-${e.code}`);
        if (keyElement) {
            keyElement.classList.add('active');
        }

        // 기능 키(Shift, Ctrl 등)는 무시 (Backspace는 예외)
        if (e.key.length > 1 && e.code !== KEY_CODE_BACKSPACE && e.code !== KEY_CODE_ENTER && e.code !== KEY_CODE_SPACE) {
            return;
        }

        // Backspace 처리
        if (e.code === KEY_CODE_BACKSPACE) {
            if (userInput.length > 0) {
                const lastChar = userInput[userInput.length - 1];
                const problemChar = currentHiraganaSentence[userInput.length - 1];
                if (lastChar === problemChar) {
                    totalCorrectChars--;
                }
                totalTypedChars--;

                userInput = userInput.slice(0, -1);
                updateDisplay();
            }
            MiddleShiftActive = false;
            updateKeyboardGuide();
            return;
        }

        // 문장이 완료되었으면 Enter 키만 허용
        if (getCorrectLength() >= currentHiraganaSentence.length) {
            if (e.code === KEY_CODE_ENTER) {
                startNewGame();
            }
            return;
        }

        // --- 입력 유효성 검사 시작 ---
        const correctLength = getCorrectLength();
        if (correctLength >= currentHiraganaSentence.length) {
             if (e.code === KEY_CODE_ENTER) startNewGame();
             return;
        }

        const nextChar = currentHiraganaSentence[correctLength];
        const requiredKeys = getKeysForChar(nextChar);
        if (requiredKeys.length === 0) { return; }

        let currentStep = 0;
        if (MiddleShiftActive) {
            currentStep = 1;
        } else {
            const baseChar = reverseDakutenMap[nextChar] || reverseHandakutenMap[nextChar];
            if (baseChar && userInput[correctLength] === baseChar) {
                currentStep = requiredKeys.length - 1;
            }
        }
        
        const expectedKeyCode = requiredKeys[currentStep].keyCode;

        if (expectedKeyCode === 'MiddleShift') {
            if (e.code !== KEY_CODE_MIDDLE_SHIFT_LEFT && e.code !== KEY_CODE_MIDDLE_SHIFT_RIGHT) {
                return; // d 또는 k가 아니면 입력 무시
            }
            // MiddleShiftActive 상태 토글
            MiddleShiftActive = !MiddleShiftActive;
            updateKeyboardGuide();
            return; // MiddleShift 활성화 후 실제 문자 입력은 다음 키 입력에서 처리
        } else {
            if (e.code !== expectedKeyCode) {
                return; // 유효하지 않은 키 입력이면 여기서 함수 종료
            }
        }
        // --- 입력 유효성 검사 끝 ---


        // 게임 타이머 시작
        if (!gameStarted && currentHiraganaSentence) {
            startGameTimer();
            gameStarted = true;
        }

        // 키 반복 방지
        if (keyPressed[e.code]) {
            return;
        }
        keyPressed[e.code] = true;

        const keyMapping = tsukiLayout[e.code];
        let typedChar = MiddleShiftActive ? keyMapping.shift : keyMapping.normal;

        // 탁음/반탁음 변환 또는 Shift + KeyL/Slash 처리
        if (e.code === KEY_CODE_DAKUTEN || e.code === KEY_CODE_HANDAKUTEN) {
            // MiddleShiftActive 상태라면, 탁음/반탁음 변환 대신 Shifted 문자 입력으로 간주
            if (MiddleShiftActive) {
                totalTypedChars++;
                if (typedChar === currentHiraganaSentence[userInput.length]) {
                    totalCorrectChars++;
                }
                userInput += typedChar;
            } else if (userInput.length > 0) {
                const prevChar = userInput[userInput.length - 1];
                const map = (e.code === KEY_CODE_DAKUTEN) ? dakutenMap : handakutenMap;
                const convertedChar = map[prevChar];
                if (convertedChar) {
                    // Adjust totalCorrectChars for conversion
                    const problemCharIndex = userInput.length - 1;
                    if (prevChar === currentHiraganaSentence[problemCharIndex]) {
                        totalCorrectChars--; // Decrement if previous char was correct
                    }
                    if (convertedChar === currentHiraganaSentence[problemCharIndex]) {
                        totalCorrectChars++; // Increment if new char is correct
                    }
                    userInput = userInput.slice(0, -1) + convertedChar;
                }
            }
        } else {
            // Normal character input
            totalTypedChars++;
            if (typedChar === currentHiraganaSentence[userInput.length]) {
                totalCorrectChars++;
            }
            userInput += typedChar;
        }

        // 일반 문자 입력 후 MiddleShiftActive 상태 재설정
        if (MiddleShiftActive) {
            MiddleShiftActive = false;
        }

        updateDisplay();
        checkCompletion();
    }
    function handleKeyUp(e) {
        const keyElement = document.getElementById(`key-${e.code}`);
        if (keyElement) {
            keyElement.classList.remove('active');
        }
        keyPressed[e.code] = false;
        updateKeyboardGuide(); // 키를 뗄 때도 가이드 업데이트
    }

    // 타이머 시작
    function startGameTimer() {
        timer = setInterval(() => {
            timeElapsed++;
            timerElement.textContent = `時間: ${timeElapsed}秒`;
            updateStats();
        }, 1000);
    }

    // 화면 업데이트
    function updateDisplay() {
        const normalChars = problemElement.querySelector('.normal-text').children;
        const hiraganaChars = problemElement.querySelector('.hiragana-text').children;

        // 1. 모든 글자의 클래스 초기화
        for (let i = 0; i < hiraganaChars.length; i++) {
            hiraganaChars[i].className = '';
        }
        for (let i = 0; i < normalChars.length; i++) {
            normalChars[i].className = '';
        }

        // 2. 입력한 부분까지 정답 표시
        for (let i = 0; i < getCorrectLength(); i++) {
            if (i < hiraganaChars.length) {
                hiraganaChars[i].classList.add('correct');
            }
        }

        // 3. 커서 위치 설정
        const cursorPos = getCorrectLength(); // 변경: userInput.length 대신 getCorrectLength() 사용
        if (cursorPos < currentHiraganaSentence.length) {
            hiraganaChars[cursorPos].classList.add('current');
        }

        // 4. 상단 문제 문장 진행상황(committed) 및 커서 업데이트
        const committedCharsCount = (getCorrectLength() > 0) ? (currentCntTypedFont[getCorrectLength() - 1] || 0) : 0;
        for (let i = 0; i < normalChars.length; i++) {
            if (i < committedCharsCount) {
                normalChars[i].classList.add('committed');
            }
        }
        if (committedCharsCount < currentNormalSentence.length) {
            normalChars[committedCharsCount].classList.add('current');
        }

        // 5. 통계 및 가이드 업데이트
        userInputDisplayElement.textContent = userInput;
        updateStats();
        updateKeyboardGuide();
    }
    
    // 통계 업데이트
    function updateStats() {
        if (timeElapsed > 0) {
            const cpm = Math.round((totalTypedChars / timeElapsed) * 60);
            cpmElement.textContent = `速度: ${cpm}字/分`;
        } else {
            cpmElement.textContent = '速度: 0字/分';
        }
    }
    
    // 완료 체크
    function checkCompletion() {
        if (getCorrectLength() >= currentHiraganaSentence.length) {
            startNewGame(); // 다음 문장으로 바로 넘어감
        }
    }
    
    // 가상 키보드 생성
    function createVirtualKeyboard() {
        const layout = [
            // Row 1: Tab, QWERTY keys
            ['KeyQ', 'KeyW', 'KeyE', 'KeyR', 'KeyT', 'KeyY', 'KeyU', 'KeyI', 'KeyO', 'KeyP', 'BracketLeft'],
            // Row 2: CapsLock, ASDF keys
            ['KeyA', 'KeyS', 'KeyD', 'KeyF', 'KeyG', 'KeyH', 'KeyJ', 'KeyK', 'KeyL', 'Semicolon', 'Quote'],
            // Row 3: Left Shift, ZXCV keys
            ['KeyZ', 'KeyX', 'KeyC', 'KeyV', 'KeyB', 'KeyN', 'KeyM', 'Comma', 'Period', 'Slash'],
        ];

        layout.forEach(row => {
            const rowDiv = document.createElement('div');
            rowDiv.className = 'keyboard-row';
            row.forEach(keyCode => {
                const keyDiv = document.createElement('div');
                keyDiv.className = 'keyboard-key';
                keyDiv.id = `key-${keyCode}`;

                const mapping = tsukiLayout[keyCode];
                if (mapping) {
                    const mainChar = document.createElement('span');
                    mainChar.className = 'key-main';
                    mainChar.textContent = mapping.normal;
                    keyDiv.appendChild(mainChar);

                    const shiftChar = document.createElement('span');
                    shiftChar.className = 'key-shift';
                    shiftChar.textContent = mapping.shift;
                    keyDiv.appendChild(shiftChar);
                }
                rowDiv.appendChild(keyDiv);
            });
            virtualKeyboardElement.appendChild(rowDiv);
        });
    }

    // 새 게임 시작 (통계 초기화 포함)
    function resetGameAndStats() {
        clearInterval(timer); // 기존 타이머 중지
        gameStarted = false;
        timeElapsed = 0;
        timerElement.textContent = '時間: 0秒';
        cpmElement.textContent = '速度: 0字/分';
        startNewGame();
    }

    // 5. 이벤트 리스너 연결
    restartBtn.addEventListener('click', resetGameAndStats);
    window.addEventListener('keydown', e => {
        const keyElement = document.getElementById(`key-${e.code}`);
        if (keyElement) {
            keyElement.classList.add('active');
        }
        handleKeyDown(e);
    });
    window.addEventListener('keyup', handleKeyUp);

    // 6. 초기화
    initializeApp();

    // 테마 관련 함수 및 초기화
    function applyTheme(theme) {
        if (theme === 'dark') {
            body.classList.add('dark-mode');
            themeToggleBtn.textContent = 'Light Mode';
        } else {
            body.classList.remove('dark-mode');
            themeToggleBtn.textContent = 'Dark Mode';
        }
        localStorage.setItem('theme', theme);
    }

    // 초기 테마 설정
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
        applyTheme(savedTheme);
    } else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
        applyTheme('dark');
    } else {
        applyTheme('light');
    }

    // 테마 토글 버튼 이벤트 리스너
    themeToggleBtn.addEventListener('click', () => {
        const currentTheme = localStorage.getItem('theme');
        if (currentTheme === 'dark') {
            applyTheme('light');
        } else {
            applyTheme('dark');
        }
    });
});
