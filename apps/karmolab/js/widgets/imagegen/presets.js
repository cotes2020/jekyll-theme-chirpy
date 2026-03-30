/**
 * imagegen - 프리셋 데이터 (컨텍스트, 캐릭터, 옵션)
 */
(function () {
    'use strict';
    window.ImageGen = window.ImageGen || {};

    const CONTEXT_PRESETS = {
        bg: [
            { id: 'ingame', icon: '🎮', label: '인게임 화면', prompt: `Anime style game screenshot, direct top-down view (90 degree overhead). Wide angle shot. HD-2D style (3D Background + Pixel Art). Setting: Inside a cozy wooden mansion. Wooden floor layout with stairs and rugs. Scattered books, magical effects. Amber-like windows. Simple and cute chibi pixel art characters (SD style) fighting. Maid Alisa sweeping blue slimes. Jiangshi Ling swinging censer. Warm sunlight, god rays. High quality.` },
            { id: 'keyVisual', icon: '🖼️', label: '키 비주얼', prompt: `Anime style game key visual illustration. Wide angle panoramic view. Interior of a cozy wooden magical mansion. Circular library room with spiral stairs. Foreground: <A> with glasses and broom. Background: <B> sleeping on a sofa. Next to her, <C>. Warm sunlight beaming down, detailed background.` },
            { id: 'lobby', icon: '🏠', label: '로비 (거실)', prompt: `Anime game background art. Wide shot of the main lobby of a wooden magical mansion. A cozy living room with a large, comfy sofa filled with messy pillows and blankets. Wooden floors, scattered magical books, and a warm fireplace. A feeling of laziness and peace. HD-2D style, bright and welcoming atmosphere.` },
            { id: 'lab', icon: '⚗️', label: '실험실', prompt: `Anime background art. Wide angle shot of a magical laboratory inside a wooden mansion. Curved wooden walls, messy bookshelves, scattered papers, alchemy flasks. Sunlight streaming through amber windows. Dust motes dancing in the light. Warm, cozy, slightly cluttered but charming atmosphere.` }
        ],
        story: [
            { id: 'ep1', icon: '🍞', label: 'Ep1. 아침', prompt: `Anime visual novel cutscene illustration, wide shot. Sweet morning atmosphere. A messy, sunlit bedroom inside a wooden mansion. <A> buried under blankets on a bed. <B> stands by the bed holding cinnamon rolls. Cinematic lighting, detailed background.` },
            { id: 'ep2', icon: '🔋', label: 'Ep2. 충전', prompt: `Anime visual novel cutscene illustration, medium shot. Intimate late-night atmosphere. <A> and <B> sitting close on a sofa. <A> sleepily leaning in, touching her forehead to <B>'s forehead. Soft blue magical glowing light. Alisa's eyes closed behind glasses. Warm mood. Cinematic lighting, detailed.` },
            { id: 'ep3', icon: '👓', label: 'Ep3. 안경', prompt: `Anime visual novel cutscene illustration, medium shot. Bright afternoon. <B> stands WITHOUT her glasses, wiping them. <A> lying on a sofa, looking at <B> with sparkling, teasing eyes. Sunlight fills the room. Cinematic lighting, high quality.` },
            { id: 'ep4', icon: '🍳', label: 'Ep4. 요리', prompt: `Anime visual novel cutscene illustration, wide shot. Comical kitchen chaos. A fantasy kitchen with a giant cauldron bubbling over with purple goo. <B> pointing angrily at a large batter stain on the ceiling. <A> looking away guiltily. Broken eggshells everywhere. Cinematic lighting.` },
            { id: 'ep5', icon: '🌙', label: 'Ep5. 악몽', prompt: `Anime visual novel cutscene illustration, close up. Emotional night scene. Dark bedroom, moonlight. <A> sitting up in bed, looking scared from nightmare. <B> holding <A>'s hand gently, looking concerned. Cinematic lighting.` },
            { id: 'ep6', icon: '❄️', label: 'Ep6. 쿨팩', prompt: `Anime visual novel cutscene illustration, medium shot. Hot summer afternoon atmosphere. <A> sweating and sleeping on a sofa. <B> hugging <A> from behind with a blissful expression. <B>'s yellow paper talisman on her forehead says 'Happy' in Kanji. Warm sunlight, detailed.` },
            { id: 'ep7', icon: '😳', label: 'Ep7. 부적', prompt: `Anime visual novel cutscene illustration, close up. <B> looking shy and blushing, trying to hide her face with a fan or hands. But the yellow paper talisman on her forehead clearly shows the Kanji for 'Love' (愛) or 'Joy' (喜). <A> is laughing in the background. Cute comedy atmosphere.` }
        ],
        lab: [
            { id: 'exp_sheep', icon: '🐏', label: '실험: 양 수인', prompt: `Character design sheet, anime style 2D illustration. <A> transformed into a Sheep Hybrid. **Curled Ram Horns (spiral shape) on the side of her head instead of earmuffs.** Messy Orange hair, half-lidded eyes, Shiba-brows, glasses. Wearing her usual loose robe. Cute sheep ears. Soft pastel colors.` },
            { id: 'exp_earmuffs', icon: '🎧', label: '실험: 귀마개만', prompt: `Character design sheet, anime style 2D illustration. <A> without her hat. **Messy Orange hair is fully visible.** Wearing **Large fluffy Sleeping Earmuffs with an ORANGE spiral pattern** directly on her ears. Half-lidded eyes, Shiba-brows, glasses. Oversized robe. Natural look.` },
            { id: 'exp_winter', icon: '❄️', label: '실험: 방한모', prompt: `Character design sheet, anime style 2D illustration. <A> wearing a **Winter Trapper Hat (Ushanka style) but with a pointy top like a witch hat.** The ear flaps covering her ears have a **distinct spiral pattern.** Fur lining. Messy Orange hair, half-lidded eyes, glasses. Oversized robe. Cozy winter vibe.` }
        ],
        mascot: [
            { id: 'm_idle', icon: '😊', label: 'idle (기본)', prompt: `Japanese anime illustration style, chibi mascot character. <CHAR>. Neutral calm expression, gentle smile, relaxed pose. Lab coat or researcher outfit. Bright science lab aesthetic. Polished Japanese illustration style, clean sharp edges. Pure white background, isolated character. For mascot/icon use.` },
            { id: 'm_happy', icon: '😄', label: 'happy (행복)', prompt: `Japanese anime illustration style, chibi mascot character. <CHAR>. Happy cheerful expression, big smile, curved happy eyes, blushing cheeks. Joyful pose. Lab coat or researcher outfit. Bright science lab aesthetic. Polished Japanese illustration style, clean sharp edges. Pure white background, isolated character. For mascot/icon use.` },
            { id: 'm_sad', icon: '😢', label: 'sad (슬픔)', prompt: `Japanese anime illustration style, chibi mascot character. <CHAR>. Sad expression, downturned mouth, teary eyes, single tear drop. Lab coat or researcher outfit. Polished Japanese illustration style, clean sharp edges. Pure white background, isolated character. For mascot/icon use.` },
            { id: 'm_shock', icon: '😲', label: 'shock (놀람)', prompt: `Japanese anime illustration style, chibi mascot character. <CHAR>. Shocked surprised expression, wide open eyes, open mouth, startled pose. Lab coat or researcher outfit. Polished Japanese illustration style, clean sharp edges. Pure white background, isolated character. For mascot/icon use.` },
            { id: 'm_think', icon: '🤔', label: 'think (생각)', prompt: `Japanese anime illustration style, chibi mascot character. <CHAR>. Thoughtful expression, eyes looking to the side, flat mouth, contemplative pose. Lab coat or researcher outfit. Polished Japanese illustration style, clean sharp edges. Pure white background, isolated character. For mascot/icon use.` },
            { id: 'm_sleep', icon: '😴', label: 'sleep (졸음)', prompt: `Japanese anime illustration style, chibi mascot character. <CHAR>. Sleeping expression, closed eyes, Zzz symbols floating above head. Peaceful relaxed pose. Lab coat or researcher outfit. Polished Japanese illustration style, clean sharp edges. Pure white background, isolated character. For mascot/icon use.` },
            { id: 'm_angry', icon: '😠', label: 'angry (화남)', prompt: `Japanese anime illustration style, chibi mascot character. <CHAR>. Angry expression, furrowed brows, angry mouth, anger vein or steam symbol. Furious pose. Lab coat or researcher outfit. Polished Japanese illustration style, clean sharp edges. Pure white background, isolated character. For mascot/icon use.` },
            { id: 'm_love', icon: '🥰', label: 'love (사랑)', prompt: `Japanese anime illustration style, chibi mascot character. <CHAR>. Love-struck expression, heart-shaped eyes, big smile, blushing cheeks, floating heart symbols. Adoring pose. Lab coat or researcher outfit. Polished Japanese illustration style, clean sharp edges. Pure white background, isolated character. For mascot/icon use.` },
            { id: 'm_smug', icon: '😏', label: 'smug (쏘옥)', prompt: `Japanese anime illustration style, chibi mascot character. <CHAR>. Smug expression, one eye winking, sly smirk, confident pose. Lab coat or researcher outfit. Polished Japanese illustration style, clean sharp edges. Pure white background, isolated character. For mascot/icon use.` },
            { id: 'm_eating', icon: '🍽️', label: 'eating (먹는 중)', prompt: `Japanese anime illustration style, chibi mascot character. <CHAR>. Eating expression, mouth with food or snack, happy eyes, eating pose. Lab coat or researcher outfit. Polished Japanese illustration style, clean sharp edges. Pure white background, isolated character. For mascot/icon use.` },
            { id: 'm_pointing', icon: '👉', label: 'pointing (가리킴)', prompt: `Japanese anime illustration style, chibi mascot character. <CHAR>. Pointing pose, one arm extended pointing forward, smile, attentive expression. Lab coat or researcher outfit. Polished Japanese illustration style, clean sharp edges. Pure white background, isolated character. For mascot/icon use.` },
            { id: 'm_cheer', icon: '🎉', label: 'cheer (환호)', prompt: `Japanese anime illustration style, chibi mascot character. <CHAR>. Excited cheering expression, star-shaped eyes, big grin, sparkles around, celebratory pose. Lab coat or researcher outfit. Polished Japanese illustration style, clean sharp edges. Pure white background, isolated character. For mascot/icon use.` }
        ],
        emoji: [
            { id: 'discord', icon: '💬', label: '디시콘 기본', prompt: 'Generate an image: <CHAR>. custom emoji sticker illustration, simple cute design, readable at small size, kawaii style, square composition, white background, single isolated character or object, 2D illustration' },
            { id: 'chibi', icon: '👤', label: '캐릭터', prompt: 'Generate an image: <CHAR>. chibi character emoji sticker, cute anime style, isolated on white background, square format, clear silhouette, mascot icon style, 2D illustration' },
            { id: 'happy', icon: '😄', label: '행복', prompt: 'Generate an image: <CHAR>. happy emoji sticker, big smile, cheerful expression, cute kawaii style, square format, white background, 2D illustration' },
            { id: 'sad', icon: '😢', label: '슬픔', prompt: 'Generate an image: <CHAR>. sad emoji sticker, teary eyes, downturned mouth, cute kawaii style, square format, white background, 2D illustration' },
            { id: 'shock', icon: '😲', label: '놀람', prompt: 'Generate an image: <CHAR>. surprised emoji sticker, wide eyes, open mouth, shocked expression, cute style, square format, white background, 2D illustration' },
            { id: 'angry', icon: '😠', label: '화남', prompt: 'Generate an image: <CHAR>. angry emoji sticker, furrowed brows, steam symbol, cute kawaii style, square format, white background, 2D illustration' },
            { id: 'love', icon: '🥰', label: '사랑', prompt: 'Generate an image: <CHAR>. love emoji sticker, heart eyes, blushing cheeks, cute kawaii style, square format, white background, 2D illustration' },
            { id: 'game', icon: '🎮', label: '게임', prompt: 'Generate an image: <CHAR>. gaming emoji sticker, controller or game themed, cute pixel or cartoon style, square format, white background, 2D illustration' },
            { id: 'cat', icon: '🐱', label: '고양이', prompt: 'Generate an image: <CHAR>. cat emoji sticker, kawaii cat face, cute simple design, square format, white background, 2D illustration' },
            { id: 'star', icon: '⭐', label: '스타', prompt: 'Generate an image: <CHAR>. star or sparkle emoji sticker, shiny cute design, square format, white or gradient background, 2D illustration' }
        ]
    };

    const CHARACTER_PRESETS = {
        char: [
            { id: 'witch', icon: '💤', label: '마녀 욘', shortLabel: 'Yawn, a young witch with messy orange hair and round glasses',
                prompt: `A young adult witch (Yawn). **Very slender body, flat chest (petite).** **Messy Orange hair.** Face: Half-open sleepy eyes (half-lidded), distinctive short thick eyebrows (maro-mayu), **wearing round glasses**, **slightly blushing cheeks (shy)**, expression of finding things troublesome but trying to hide it. **Headwear: Drooping Nightcap (sleeping hat). Accessories: Large fluffy sleeping earmuffs with an ORANGE spiral pattern.** Outfit: Oversized loose fitting witch robe falling off shoulder. Introverted and cute atmosphere, soft colors.` },
            { id: 'alisa', icon: '🧹', label: '메이드 알리사', shortLabel: 'Alisa, a maid with glasses and black ponytail',
                prompt: `A cute maid (Alisa). Face: Sharp intellectual eyes, stylish glasses (megane), stoic cool beauty expression. Black ponytail. Wearing a classic black and white maid outfit. Holding a large magical broomstick. Dynamic posing. Clean background, detailed.` },
            { id: 'ling', icon: '🧟‍♀️', label: '강시 링', shortLabel: 'Ling, a Jiangshi vampire maid girl',
                prompt: `A beautiful Jiangshi (Chinese vampire) maid girl named Ling. Face: Innocent baby face, mischievous smile. Body: Glamorous and curvy. Dark brown hair in cute twin-buns. Costume: Black Qipao-Maid fusion dress, form-fitting with frills. Paper talisman on forehead. Floating pose. White background, detailed.` },
            { id: 'timeto', icon: '💜', label: '티메토', shortLabel: 'Timeto, young girl with lavender hair',
                prompt: `Timeto (티메토) — young girl lab director with lavender/purple hair` }
        ]
    };

    // Optional SSoT (wiki MD → load-characters-from-wiki): 동일 preset id만 덮어쓰고 나머지(예: timeto) 유지
    try {
        const b = window.KarmoWorld?.bindings?.imagegen?.characters;
        if (Array.isArray(b) && b.length) {
            const byId = new Map(CHARACTER_PRESETS.char.map(row => [row.id, { ...row }]));
            b.forEach(x => {
                if (!x.imagegenPresetId || !x.prompt) return;
                byId.set(x.imagegenPresetId, {
                    id: x.imagegenPresetId,
                    icon: x.icon,
                    label: x.label,
                    shortLabel: x.shortLabel || '',
                    prompt: x.prompt || ''
                });
            });
            CHARACTER_PRESETS.char = Array.from(byId.values());
        }
    } catch (_) {}

    const CUSTOM_INPUT_ID = '_custom';
    const CUSTOM_PRESETS_KEY = 'toolbox_imagegen_custom_presets';
    const CUSTOM_CHARACTERS_KEY = 'toolbox_imagegen_custom_characters';

    const CONTEXT_TAB_LABELS = { bg: '배경', story: '스토리', lab: '실험실', mascot: '🐱 마스코트', emoji: '😀 이모지', custom: '⭐ 커스텀' };
    const CONTEXT_TAB_ICONS = { bg: '🖼️', story: '📖', lab: '⚗️', mascot: '🐱', emoji: '😀', custom: '⭐' };

    const VIBE_OPTIONS = [
        { id: 'none', label: '없음', suffix: '', desc: '추가 스타일 없이 프롬프트 그대로 생성합니다.' },
        { id: 'cute', label: '🧸 Cute', suffix: ', cute adorable kawaii pastel colors soft lighting', desc: '파스텔 톤, 부드러운 조명의 귀엽고 사랑스러운 스타일.' },
        { id: 'pure', label: '✨ Pure', suffix: ', clean pure white aesthetic, minimal, elegant, soft, ethereal lighting', desc: '깨끗한 흰색 배경, 미니멀하고 우아한 분위기.' },
        { id: 'dramatic', label: '⚡ Dramatic', suffix: ', dramatic dynamic pose, bold vibrant colors, cinematic lighting, high contrast', desc: '역동적인 포즈, 선명한 색상, 시네마틱 조명.' },
        { id: 'spicy', label: '🔥 Spicy', suffix: ', alluring romantic atmosphere, warm passionate lighting, captivating mood, smoldering gaze', desc: '매혹적이고 로맨틱한 분위기, 따뜻한 조명.' },
        { id: 'dark', label: '🌑 Dark', suffix: ', dark moody atmosphere, gothic, muted colors, dramatic shadows', desc: '어두운 고딕 분위기, 깊은 그림자와 음울한 색감.' },
        { id: 'retro', label: '📺 Retro', suffix: ', 80s retro anime style, VHS aesthetic, warm vintage color palette, film grain', desc: '80년대 레트로 애니메 느낌, VHS 질감과 빈티지 색감.' },
        { id: 'pixel', label: '👾 Pixel', suffix: ', pixel art style, 16-bit retro game sprite, limited color palette', desc: '16비트 레트로 게임 스타일의 픽셀 아트.' }
    ];

    const ASPECT_RATIOS = [
        { value: '16:9', label: '16:9 (가로)' },
        { value: '1:1', label: '1:1 (정사각형)' },
        { value: '9:16', label: '9:16 (세로)' },
        { value: '3:4', label: '3:4 (세로)' },
        { value: '4:3', label: '4:3 (가로)' },
        { value: '3:2', label: '3:2 (가로)' },
        { value: '2:3', label: '2:3 (세로)' }
    ];

    const SAFETY_LEVELS = [
        { value: 'OFF', label: '끄기 (OFF)' },
        { value: 'BLOCK_NONE', label: '차단 없음 (BLOCK_NONE)' },
        { value: 'BLOCK_ONLY_HIGH', label: '높음만 차단 (BLOCK_ONLY_HIGH)' },
        { value: 'BLOCK_MEDIUM_AND_ABOVE', label: '중간 이상 차단 (BLOCK_MEDIUM_AND_ABOVE)' },
        { value: 'BLOCK_LOW_AND_ABOVE', label: '낮음 이상 차단 (BLOCK_LOW_AND_ABOVE)' }
    ];

    const PERSON_GEN_OPTIONS = [
        { value: 'allow_adult', label: '성인만 허용 (기본)' },
        { value: 'allow_all', label: '모두 허용' },
        { value: 'dont_allow', label: '비허용' }
    ];

    function getSlotsFromPrompt(prompt) {
        const m = prompt.match(/<([A-Z]+)>/g);
        return m ? [...new Set(m.map(s => s.slice(1, -1)))] : [];
    }

    function loadCustomCharacters() {
        try { return JSON.parse(localStorage.getItem(CUSTOM_CHARACTERS_KEY)) || []; }
        catch (_) { return []; }
    }

    function saveCustomCharacters(list) {
        localStorage.setItem(CUSTOM_CHARACTERS_KEY, JSON.stringify(list));
    }

    function getCharacterOptions() {
        const builtin = CHARACTER_PRESETS.char || [];
        const custom = loadCustomCharacters();
        return [...builtin, ...custom];
    }

    function loadCustomPresets() {
        try { return JSON.parse(localStorage.getItem(CUSTOM_PRESETS_KEY)) || []; }
        catch (_) { return []; }
    }

    function saveCustomPresets(list) {
        localStorage.setItem(CUSTOM_PRESETS_KEY, JSON.stringify(list));
    }

    Object.assign(window.ImageGen, {
        CONTEXT_PRESETS,
        CHARACTER_PRESETS,
        CUSTOM_INPUT_ID,
        CUSTOM_PRESETS_KEY,
        CUSTOM_CHARACTERS_KEY,
        CONTEXT_TAB_LABELS,
        CONTEXT_TAB_ICONS,
        VIBE_OPTIONS,
        ASPECT_RATIOS,
        SAFETY_LEVELS,
        PERSON_GEN_OPTIONS,
        getSlotsFromPrompt,
        loadCustomCharacters,
        saveCustomCharacters,
        getCharacterOptions,
        loadCustomPresets,
        saveCustomPresets
    });
})();
