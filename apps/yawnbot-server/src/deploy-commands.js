/**
 * 슬래시 커맨드 등록 (Discord API에 등록하는 스크립트)
 * 사용법: node src/deploy-commands.js
 */
require('dotenv').config();
const { REST, Routes, SlashCommandBuilder } = require('discord.js');

const commands = [
    // ── 강화 ──
    new SlashCommandBuilder().setName('강화').setDescription('검을 강화합니다. (확률 존재)'),
    new SlashCommandBuilder().setName('판매').setDescription('검을 판매하여 돈을 얻습니다.'),
    new SlashCommandBuilder().setName('정보').setDescription('내 검과 재산 정보를 확인합니다.'),
    new SlashCommandBuilder().setName('돈').setDescription('현재 보유한 돈을 확인합니다.'),
    new SlashCommandBuilder().setName('랭킹').setDescription('전체 유저 랭킹을 확인합니다.'),
    new SlashCommandBuilder().setName('출첵').setDescription('매일 출석체크 보상을 받습니다.'),
    new SlashCommandBuilder().setName('돈내놔').setDescription('일정 시간마다 랜덤 용돈을 받습니다.'),

    // ── 미니게임 ──
    new SlashCommandBuilder().setName('배틀').setDescription('다른 유저와 대결합니다.')
        .addUserOption(opt => opt.setName('상대').setDescription('대결할 상대를 선택하세요').setRequired(true)),
    new SlashCommandBuilder().setName('슬롯').setDescription('슬롯 머신을 돌립니다.')
        .addIntegerOption(opt => opt.setName('금액').setDescription('배팅할 금액').setRequired(true).setMinValue(1)),
    new SlashCommandBuilder().setName('홀짝').setDescription('홀짝 게임을 합니다.')
        .addStringOption(opt => opt.setName('선택').setDescription('홀 또는 짝').setRequired(true)
            .addChoices({ name: '홀', value: '홀' }, { name: '짝', value: '짝' }))
        .addIntegerOption(opt => opt.setName('금액').setDescription('배팅할 금액').setRequired(true).setMinValue(1)),
    new SlashCommandBuilder().setName('가위바위보').setDescription('가위바위보를 합니다.')
        .addStringOption(opt => opt.setName('선택').setDescription('가위, 바위, 보').setRequired(true)
            .addChoices({ name: '가위', value: '가위' }, { name: '바위', value: '바위' }, { name: '보', value: '보' }))
        .addIntegerOption(opt => opt.setName('금액').setDescription('배팅할 금액').setRequired(true).setMinValue(1)),

    // ── 주식 ──
    new SlashCommandBuilder().setName('주식목록').setDescription('현재 상장된 주식 시세를 확인합니다.'),
    new SlashCommandBuilder().setName('매수').setDescription('주식을 매수합니다.')
        .addStringOption(opt => opt.setName('종목').setDescription('종목 심볼').setRequired(true)
            .addChoices(
                { name: '떡락전자 (SAMSUNG)', value: 'SAMSUNG' },
                { name: '화성갈끄니까 (DOGE)', value: 'DOGE' },
                { name: '테슬라 (TESLA)', value: 'TESLA' },
                { name: '사과 (APPLE)', value: 'APPLE' },
                { name: '비트코인 (BITCOIN)', value: 'BITCOIN' },
            ))
        .addIntegerOption(opt => opt.setName('수량').setDescription('매수할 수량').setRequired(true).setMinValue(1)),
    new SlashCommandBuilder().setName('매도').setDescription('주식을 매도합니다.')
        .addStringOption(opt => opt.setName('종목').setDescription('종목 심볼').setRequired(true)
            .addChoices(
                { name: '떡락전자 (SAMSUNG)', value: 'SAMSUNG' },
                { name: '화성갈끄니까 (DOGE)', value: 'DOGE' },
                { name: '테슬라 (TESLA)', value: 'TESLA' },
                { name: '사과 (APPLE)', value: 'APPLE' },
                { name: '비트코인 (BITCOIN)', value: 'BITCOIN' },
            ))
        .addIntegerOption(opt => opt.setName('수량').setDescription('매도할 수량').setRequired(true).setMinValue(1)),
    new SlashCommandBuilder().setName('내주식').setDescription('내 주식 잔고를 확인합니다.'),

    // ── 레이드 ──
    new SlashCommandBuilder().setName('레이드정보').setDescription('현재 진행 중인 레이드 정보를 확인합니다.'),
    new SlashCommandBuilder().setName('공격').setDescription('레이드 보스를 공격합니다.'),
    new SlashCommandBuilder().setName('레이드소환').setDescription('새로운 레이드 보스를 소환합니다.'),

    // ── 일반 ──
    new SlashCommandBuilder().setName('ping').setDescription('봇의 응답 속도를 확인합니다.'),
    new SlashCommandBuilder().setName('도움말').setDescription('도움말을 확인합니다.'),

    // ── AI ──
    new SlashCommandBuilder().setName('yawn').setDescription('Gemini AI에게 무엇이든 물어보세요!')
        .addStringOption(opt => opt.setName('질문').setDescription('AI에게 전달할 메시지').setRequired(true)),

    // ── 관리자 ──
    new SlashCommandBuilder().setName('admin-reload').setDescription('[관리자] 데이터를 다시 불러옵니다.'),
    new SlashCommandBuilder().setName('admin-save').setDescription('[관리자] 데이터를 저장합니다.'),
].map(cmd => cmd.toJSON());

async function deploy() {
    const rest = new REST().setToken(process.env.DISCORD_TOKEN);
    try {
        console.log(`[Deploy] ${commands.length}개 커맨드 등록 중...`);
        await rest.put(Routes.applicationCommands(process.env.CLIENT_ID), { body: commands });
        console.log('[Deploy] 커맨드 등록 완료!');
    } catch (err) {
        console.error('[Deploy] 오류:', err);
    }
}

deploy();
