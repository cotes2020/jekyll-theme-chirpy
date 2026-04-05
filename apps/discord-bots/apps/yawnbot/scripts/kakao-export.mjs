/**
 * 카카오톡 PC 대화 저장 + (기본) 제미니 요약·디스코드 웹훅 — 단일 스크립트.
 *
 * 기본: 트리거(저장) 후 이번에 생긴 .txt 만 스캔해 요약
 *   npm run kakao-export
 *   npm run kakao-export -- --once
 *   npm run kakao-export -- --countdown 0
 *
 * 모드:
 *   --watch              폴더 상시 감시만 (백그라운드용과 동일 동작)
 *   --scan [epochMs]     .txt 한 번만 처리 (생략 시 전체, 숫자면 mtime 기준)
 *   --trigger-only       저장만 하고 요약 안 함 (테스트: npm run kakao-export-save). 후보 없음이면 스냅샷 직후 exit 0(카운트다운·저장 PS 생략)
 *
 * 환경 변수: KAKAO_EXPORT_WATCH_DIR, GEMINI_API_KEY, GEMINI_MODEL,
 *   DISCORD_SUMMARY_WEBHOOK_URL, KAKAO_EXPORT_MAX_ROUNDS,
 *   KAKAO_EXPORT_SKIP_WINDOW_TITLES — 제외할 창 제목(정확 일치·대소문자 무시·| 구분). 남는 창이 전부 스킵이면 창 없음으로 종료(메인에 Ctrl+S 안 보냄). 비우려면 OFF
 *   KAKAO_EXPORT_SAVE_WAIT_TIMEOUT_SEC, KAKAO_EXPORT_SAVE_POLL_MS, KAKAO_EXPORT_AFTER_FILE_MS,
 *   KAKAO_EXPORT_FILE_STABLE_CHECKS — 파일 크기가 같은 연속 확인 횟수(기본 3)
 */

import { spawnSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import os from 'os';
import crypto from 'crypto';
import { fileURLToPath } from 'url';
import { config } from 'dotenv';
import { setTimeout as delay } from 'node:timers/promises';
import { GoogleGenerativeAI } from '@google/generative-ai';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
config({ path: path.join(__dirname, '..', '.env') });

const DEFAULT_EXPORT_DIR = path.join(os.homedir(), 'Documents', '카카오톡 받은 파일');
const WATCH_DIR = process.env.KAKAO_EXPORT_WATCH_DIR?.trim() || DEFAULT_EXPORT_DIR;
const WEBHOOK = process.env.DISCORD_SUMMARY_WEBHOOK_URL?.trim() || '';
const MODEL = process.env.GEMINI_MODEL?.trim() || 'gemini-2.5-flash';

const STATE_ROOT = path.join(os.homedir(), '.karmolab');
const watchKey = crypto.createHash('sha256').update(WATCH_DIR).digest('hex').slice(0, 16);
const LAST_FULL_PATH = path.join(STATE_ROOT, 'kakao-export', `${watchKey}-last-full.txt`);

const PIPELINE_SINCE_SLACK_MS = 10_000;

// ─── args ──────────────────────────────────────────────────────────

function partitionArgs(argv) {
  const args = argv.slice(2);
  let mode = 'pipeline';
  let scanSinceMs = 0;
  const trigger = [];

  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === '--watch') {
      mode = 'watch';
      continue;
    }
    if (a === '--trigger-only') {
      mode = 'trigger-only';
      continue;
    }
    if (a === '--scan') {
      mode = 'scan';
      if (args[i + 1] != null && /^\d+$/.test(args[i + 1])) {
        scanSinceMs = parseInt(args[++i], 10);
      }
      continue;
    }
    trigger.push(a);
  }
  return { mode, scanSinceMs, triggerArgs: trigger };
}

function parseTriggerArgs(triggerArgs) {
  let countdown = 2;
  let saveDialogOnly = false;
  let once = false;
  for (let i = 0; i < triggerArgs.length; i++) {
    const a = triggerArgs[i];
    if (a === '--countdown' && triggerArgs[i + 1] != null) {
      countdown = Math.max(0, parseInt(triggerArgs[++i], 10) || 0);
    } else if (a === '--save-dialog-only') {
      saveDialogOnly = true;
    } else if (a === '--once') {
      once = true;
    }
  }
  return { countdown, saveDialogOnly, once };
}

function intEnv(name, def) {
  const v = parseInt(String(process.env[name] || '').trim(), 10);
  return Number.isFinite(v) && v >= 0 ? v : def;
}

function pad2(n) {
  return String(n).padStart(2, '0');
}

function exportFileName(roundIndex) {
  const d = new Date();
  const stamp =
    `${d.getFullYear()}${pad2(d.getMonth() + 1)}${pad2(d.getDate())}-` +
    `${pad2(d.getHours())}${pad2(d.getMinutes())}${pad2(d.getSeconds())}-` +
    `${String(d.getMilliseconds()).padStart(3, '0')}`;
  return `${stamp}-r${roundIndex}-kakao.txt`;
}

// ─── PowerShell trigger ────────────────────────────────────────────

function runPowerShellBlock(psBody) {
  const utf16 = Buffer.from(psBody, 'utf16le');
  const encoded = utf16.toString('base64');
  const r = spawnSync(
    'powershell.exe',
    ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-EncodedCommand', encoded],
    {
      encoding: 'utf8',
      env: process.env,
      windowsHide: true,
    },
  );
  if (r.error) throw r.error;
  const errText = (r.stderr || '').trim();
  const outText = (r.stdout || '').trim();
  if (r.status !== 0) {
    throw new Error(errText || outText || `powershell exit ${r.status}`);
  }
  return outText;
}

/** PS 에 넘길 제목 스킵: 메인 창 제목과 정확히 같으면 후보에서 제외. OFF = 스킵 없음 */
function applySkipWindowTitlesEnv() {
  const raw = process.env.KAKAO_EXPORT_SKIP_WINDOW_TITLES;
  if (raw === undefined || String(raw).trim() === '') {
    process.env.KAKAO_EXPORT_SKIP_WINDOW_TITLES = '카카오톡|KakaoTalk';
    return;
  }
  const t = String(raw).trim();
  if (t === 'OFF' || t === '0') {
    process.env.KAKAO_EXPORT_SKIP_WINDOW_TITLES = '';
  }
}

const KAKAO_PICK_CS = `
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

public class KakaoPick {
  public delegate bool EnumProc(IntPtr hWnd, IntPtr lParam);

  [DllImport("user32.dll")] public static extern bool EnumWindows(EnumProc lpEnumFunc, IntPtr lParam);
  [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint processId);
  [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr hWnd);
  [DllImport("user32.dll", CharSet = CharSet.Unicode)]
  public static extern int GetWindowText(IntPtr hWnd, StringBuilder lpString, int nMaxCount);
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);
  [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);

  public static void Focus(IntPtr h) {
    ShowWindow(h, 9);
    SetForegroundWindow(h);
  }

  public static IntPtr PickWindow(int[] processIds, string[] skipTitlesExactIgnoreCase) {
    var idSet = new HashSet<uint>();
    foreach (var p in processIds) idSet.Add((uint)p);
    var list = new List<Tuple<IntPtr, string>>();
    EnumWindows((h, l) => {
      uint pid;
      GetWindowThreadProcessId(h, out pid);
      if (!idSet.Contains(pid)) return true;
      if (!IsWindowVisible(h)) return true;
      var sb = new StringBuilder(1024);
      int n = GetWindowText(h, sb, 1024);
      string title = n > 0 ? sb.ToString() : "";
      if (string.IsNullOrWhiteSpace(title)) return true;
      list.Add(Tuple.Create(h, title));
      return true;
    }, IntPtr.Zero);

    list.Sort((a, b) => string.Compare(a.Item2, b.Item2, StringComparison.Ordinal));

    if (skipTitlesExactIgnoreCase != null && skipTitlesExactIgnoreCase.Length > 0) {
      foreach (var t in list) {
        bool skip = false;
        string ti = t.Item2.Trim();
        foreach (var s in skipTitlesExactIgnoreCase) {
          if (string.IsNullOrEmpty(s)) continue;
          if (string.Equals(ti, s.Trim(), StringComparison.OrdinalIgnoreCase)) { skip = true; break; }
        }
        if (!skip) return t.Item1;
      }
      return IntPtr.Zero;
    }
    return list.Count > 0 ? list[0].Item1 : IntPtr.Zero;
  }

  public static bool HasPickCandidate(int[] processIds, string[] skipTitlesExactIgnoreCase) {
    return PickWindow(processIds, skipTitlesExactIgnoreCase) != IntPtr.Zero;
  }

  public static string FormatVisibleWindowsForLog(int[] processIds, string[] skipTitlesExactIgnoreCase) {
    var idSet = new HashSet<uint>();
    foreach (var p in processIds) idSet.Add((uint)p);
    var list = new List<Tuple<IntPtr, string>>();
    EnumWindows((h, l) => {
      uint pid;
      GetWindowThreadProcessId(h, out pid);
      if (!idSet.Contains(pid)) return true;
      if (!IsWindowVisible(h)) return true;
      var sb = new StringBuilder(1024);
      int n = GetWindowText(h, sb, 1024);
      string title = n > 0 ? sb.ToString() : "";
      if (string.IsNullOrWhiteSpace(title)) return true;
      list.Add(Tuple.Create(h, title));
      return true;
    }, IntPtr.Zero);
    list.Sort((a, b) => string.Compare(a.Item2, b.Item2, StringComparison.Ordinal));
    var outSb = new StringBuilder();
    foreach (var t in list) {
      string ti = t.Item2.Trim().Replace("\\r", " ").Replace("\\n", " ").Replace("\\t", " ");
      bool skip = false;
      if (skipTitlesExactIgnoreCase != null) {
        foreach (var s in skipTitlesExactIgnoreCase) {
          if (string.IsNullOrEmpty(s)) continue;
          if (string.Equals(ti, s.Trim(), StringComparison.OrdinalIgnoreCase)) { skip = true; break; }
        }
      }
      outSb.Append(skip ? "[스킵]" : "[후보]");
      outSb.Append("\\t");
      outSb.Append(ti);
      outSb.Append("\\t");
      outSb.AppendFormat("hwnd=0x{0:X}", t.Item1.ToInt64());
      outSb.Append("\\r\\n");
    }
    return outSb.Length == 0 ? "(제목 있는 보이는 KakaoTalk 창 없음)" : outSb.ToString().TrimEnd();
  }
}
`.trim();

/** 트리거 직전: 프로세스 요약 + 보이는 창 목록. 마지막 줄 PICK_HAS_CANDIDATE:1|0 (PickWindow 와 동일). @returns 후보 창 있으면 true */
function logKakaoProcessesOnce() {
  const ps = `
$ProgressPreference = 'SilentlyContinue'
$ErrorActionPreference = 'SilentlyContinue'
try {
  [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
  $OutputEncoding = [System.Text.UTF8Encoding]::new($false)
} catch { }
Write-Output '--- Get-Process (프로세스당 MainWindowTitle 1개뿐 — 여러 채팅창 대표 아님) ---'
$exact = @(Get-Process -Name 'KakaoTalk' -ErrorAction SilentlyContinue)
if ($exact.Count -gt 0) {
  foreach ($e in $exact) {
    $t = if ($e.MainWindowTitle) { ($e.MainWindowTitle -replace '\\s+', ' ').Trim() } else { '' }
    Write-Output ('EXACT:KakaoTalk PID=' + $e.Id + ' MainWindowTitle="' + $t + '"')
  }
} else {
  Write-Output 'EXACT:KakaoTalk (없음 — 스크립트는 Get-Process -Name KakaoTalk 사용)'
}
$fuzzy = @(Get-Process | Where-Object { $_.Name -match '(?i)kakao' })
if ($fuzzy.Count -eq 0) {
  Write-Output 'FUZZY:(이름에 kakao 포함 — 없음)'
} else {
  $fuzzy | Sort-Object Name, Id | ForEach-Object {
    $t = if ($_.MainWindowTitle) { ($_.MainWindowTitle -replace '\\s+', ' ').Trim() } else { '' }
    Write-Output ('FUZZY:' + $_.Name + ' PID=' + $_.Id + ' MainWindowTitle="' + $t + '"')
  }
}
$skipShow = if ($env:KAKAO_EXPORT_SKIP_WINDOW_TITLES) { $env:KAKAO_EXPORT_SKIP_WINDOW_TITLES } else { '(기본값 적용 중)' }
Write-Output ('--- 보이는 창 전부 (PickWindow 동일, 스킵=' + $skipShow + ', 제목 정확 일치만 [스킵]) ---')
Add-Type @'
${KAKAO_PICK_CS}
'@
$pids = @(Get-Process -Name 'KakaoTalk' -ErrorAction SilentlyContinue | ForEach-Object { $_.Id } | Select-Object -Unique)
$skipArr = @()
if ($env:KAKAO_EXPORT_SKIP_WINDOW_TITLES) {
  $skipArr = $env:KAKAO_EXPORT_SKIP_WINDOW_TITLES -split '\\|' | ForEach-Object { $_.Trim() } | Where-Object { $_ }
}
if ($pids.Length -eq 0) {
  Write-Output '(KakaoTalk 없어 창 목록 생략)'
} else {
  $block = [KakaoPick]::FormatVisibleWindowsForLog([int[]]@($pids), [string[]]@($skipArr))
  foreach ($line in [regex]::Split($block, '\\r\\n|\\r|\\n')) { if ($line) { Write-Output $line } }
}
$pickOk = $false
if ($pids.Length -gt 0) { $pickOk = [KakaoPick]::HasPickCandidate([int[]]@($pids), [string[]]@($skipArr)) }
Write-Output ('PICK_HAS_CANDIDATE:' + $(if ($pickOk) { '1' } else { '0' }))
exit 0
`.trim();
  const utf16 = Buffer.from(ps, 'utf16le');
  const encoded = utf16.toString('base64');
  const r = spawnSync('powershell.exe', ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-EncodedCommand', encoded], {
    encoding: 'utf8',
    windowsHide: true,
    env: process.env,
  });
  console.log('[proc] Kakao 스냅샷:');
  const out = (r.stdout || '').trim();
  let pickLineSeen = false;
  let hasPickCandidate = true;
  if (out) {
    for (const line of out.split(/\r?\n/)) {
      const t = line.trim();
      const m = /^PICK_HAS_CANDIDATE:(0|1)$/.exec(t);
      if (m) {
        pickLineSeen = true;
        hasPickCandidate = m[1] === '1';
        continue;
      }
      if (t) console.log('  ', line);
    }
  } else {
    console.log('  (PowerShell 출력 없음)');
  }
  const err = (r.stderr || '').trim();
  if (err && !/^#<\s*CLIXML/i.test(err) && !/<Objs\s+Version=/i.test(err)) {
    console.log('  [proc stderr]', err);
  }
  if (r.status !== 0 && r.status != null) console.log('  [proc] exit', r.status);
  return pickLineSeen ? hasPickCandidate : true;
}

function buildScriptSaveThroughEnter() {
  return `
$ProgressPreference = 'SilentlyContinue'
$ErrorActionPreference = 'Stop'
Add-Type @'
${KAKAO_PICK_CS}
'@
$path = $env:KAKAO_EXPORT_OUT_PATH
if (-not $path) { throw 'KAKAO_EXPORT_OUT_PATH missing' }
$pids = @(Get-Process -Name 'KakaoTalk' -ErrorAction SilentlyContinue | ForEach-Object { $_.Id } | Select-Object -Unique)
if ($pids.Length -eq 0) { throw 'KakaoTalk process not found' }
$skipArr = @()
if ($env:KAKAO_EXPORT_SKIP_WINDOW_TITLES) {
  $skipArr = $env:KAKAO_EXPORT_SKIP_WINDOW_TITLES -split '\\|' | ForEach-Object { $_.Trim() } | Where-Object { $_ }
}
$hwnd = [KakaoPick]::PickWindow([int[]]@($pids), [string[]]@($skipArr))
if ($hwnd -eq [IntPtr]::Zero) { throw 'KakaoTalk window not found' }
[KakaoPick]::Focus($hwnd)
Start-Sleep -Milliseconds 400
Add-Type -AssemblyName System.Windows.Forms
[System.Windows.Forms.SendKeys]::SendWait('^s')
Start-Sleep -Milliseconds 900
Set-Clipboard -Value $path
Start-Sleep -Milliseconds 250
[System.Windows.Forms.SendKeys]::SendWait('^a')
[System.Windows.Forms.SendKeys]::SendWait('^v')
Start-Sleep -Milliseconds 120
[System.Windows.Forms.SendKeys]::SendWait('{ENTER}')
'OK'
`.trim();
}

function buildScriptEscTwice() {
  return `
$ProgressPreference = 'SilentlyContinue'
$ErrorActionPreference = 'Stop'
Add-Type @'
${KAKAO_PICK_CS}
'@
$pids = @(Get-Process -Name 'KakaoTalk' -ErrorAction SilentlyContinue | ForEach-Object { $_.Id } | Select-Object -Unique)
if ($pids.Length -eq 0) { throw 'KakaoTalk process not found' }
$skipArr = @()
if ($env:KAKAO_EXPORT_SKIP_WINDOW_TITLES) {
  $skipArr = $env:KAKAO_EXPORT_SKIP_WINDOW_TITLES -split '\\|' | ForEach-Object { $_.Trim() } | Where-Object { $_ }
}
$hwnd = [KakaoPick]::PickWindow([int[]]@($pids), [string[]]@($skipArr))
if ($hwnd -eq [IntPtr]::Zero) { throw 'KakaoTalk window not found' }
[KakaoPick]::Focus($hwnd)
Start-Sleep -Milliseconds 300
Add-Type -AssemblyName System.Windows.Forms
[System.Windows.Forms.SendKeys]::SendWait('{ESC}')
Start-Sleep -Milliseconds 280
[System.Windows.Forms.SendKeys]::SendWait('{ESC}')
'OK'
`.trim();
}

function buildScriptSaveDialogOnly() {
  return `
$ProgressPreference = 'SilentlyContinue'
$ErrorActionPreference = 'Stop'
$path = $env:KAKAO_EXPORT_OUT_PATH
if (-not $path) { throw 'KAKAO_EXPORT_OUT_PATH missing' }
Add-Type @'
${KAKAO_PICK_CS}
'@
$pids = @(Get-Process -Name 'KakaoTalk' -ErrorAction SilentlyContinue | ForEach-Object { $_.Id } | Select-Object -Unique)
if ($pids.Length -eq 0) { throw 'KakaoTalk process not found' }
$skipArr = @()
if ($env:KAKAO_EXPORT_SKIP_WINDOW_TITLES) {
  $skipArr = $env:KAKAO_EXPORT_SKIP_WINDOW_TITLES -split '\\|' | ForEach-Object { $_.Trim() } | Where-Object { $_ }
}
$hwnd = [KakaoPick]::PickWindow([int[]]@($pids), [string[]]@($skipArr))
if ($hwnd -eq [IntPtr]::Zero) { throw 'KakaoTalk window not found' }
[KakaoPick]::Focus($hwnd)
Start-Sleep -Milliseconds 200
Add-Type -AssemblyName System.Windows.Forms
Set-Clipboard -Value $path
Start-Sleep -Milliseconds 200
[System.Windows.Forms.SendKeys]::SendWait('^a')
[System.Windows.Forms.SendKeys]::SendWait('^v')
Start-Sleep -Milliseconds 120
[System.Windows.Forms.SendKeys]::SendWait('{ENTER}')
'OK'
`.trim();
}

/** 저장 Enter 후 디스크에 파일이 생기고 크기가 잠시 안정될 때까지 (Node에서 폴링) */
async function waitForExportedFile(outPath) {
  const timeoutMs = Math.max(3000, intEnv('KAKAO_EXPORT_SAVE_WAIT_TIMEOUT_SEC', 90) * 1000);
  const pollMs = Math.max(50, intEnv('KAKAO_EXPORT_SAVE_POLL_MS', 200));
  const stableNeeded = Math.max(2, intEnv('KAKAO_EXPORT_FILE_STABLE_CHECKS', 3));
  const afterFileMs = Math.max(
    0,
    intEnv('KAKAO_EXPORT_AFTER_FILE_MS', intEnv('KAKAO_EXPORT_AFTER_SAVE_MS', 400)),
  );

  const start = Date.now();
  let lastSize = -1;
  let stableCount = 0;

  while (Date.now() - start < timeoutMs) {
    try {
      const st = fs.statSync(outPath);
      if (st.size > 0 && st.size === lastSize) {
        stableCount++;
        if (stableCount >= stableNeeded) {
          if (afterFileMs > 0) await delay(afterFileMs);
          console.error('[wait] 파일 안정:', outPath, `(${st.size} bytes)`);
          return;
        }
      } else {
        stableCount = st.size > 0 ? 1 : 0;
        lastSize = st.size;
      }
    } catch {
      stableCount = 0;
      lastSize = -1;
    }
    await delay(pollMs);
  }

  throw new Error(`Save timeout (${Math.round(timeoutMs / 1000)}s): ${outPath}`);
}

function isNoWindowError(msg) {
  return /KakaoTalk (window|process) not found/i.test(String(msg || ''));
}

const MSG_NO_PICKABLE_CHAT =
  '저장할 채팅 창이 없습니다. 메인만 열려 있거나, 보이는 창 제목이 모두 KAKAO_EXPORT_SKIP_WINDOW_TITLES 와 일치합니다.';

async function runTrigger({ countdown, saveDialogOnly, once }) {
  const maxRounds = Math.max(1, intEnv('KAKAO_EXPORT_MAX_ROUNDS', 80));
  const betweenMs = intEnv('KAKAO_EXPORT_BETWEEN_CHATS_MS', 500);

  fs.mkdirSync(WATCH_DIR, { recursive: true });
  applySkipWindowTitlesEnv();
  const hasPickCandidate = logKakaoProcessesOnce();
  if (!hasPickCandidate) {
    console.error(MSG_NO_PICKABLE_CHAT);
    process.exit(0);
  }

  console.error('※ 저장 파일이 디스크에 안정된 뒤 Esc Esc (Node 폴링). 메인 제목 스킵:', process.env.KAKAO_EXPORT_SKIP_WINDOW_TITLES || '(없음)');
  console.error('※ 채팅 창만 남기면, 남은 창이 없을 때까지 반복.');
  console.error(`※ 트리거: ${saveDialogOnly ? '저장창만 1회' : once ? '한 창만' : '전부 순차'}`);

  for (let i = countdown; i > 0; i--) {
    console.error(`대기 ${i}…`);
    await delay(1000);
  }

  if (saveDialogOnly) {
    const outPath = path.join(WATCH_DIR, exportFileName(1));
    process.env.KAKAO_EXPORT_OUT_PATH = outPath;
    try {
      runPowerShellBlock(buildScriptSaveDialogOnly());
    } catch (e) {
      if (isNoWindowError(e.message)) {
        console.error(MSG_NO_PICKABLE_CHAT);
        process.exit(0);
      }
      throw e;
    }
    await waitForExportedFile(outPath);
    console.error('저장 완료:', outPath);
    return;
  }

  let round = 0;
  while (true) {
    round++;
    if (round > maxRounds) {
      console.error(`KAKAO_EXPORT_MAX_ROUNDS(${maxRounds}) 도달, 중단.`);
      process.exit(1);
    }

    const outPath = path.join(WATCH_DIR, exportFileName(round));
    process.env.KAKAO_EXPORT_OUT_PATH = outPath;

    try {
      runPowerShellBlock(buildScriptSaveThroughEnter());
      await waitForExportedFile(outPath);
      runPowerShellBlock(buildScriptEscTwice());
    } catch (e) {
      if (isNoWindowError(e.message)) {
        if (round === 1) {
          console.error(MSG_NO_PICKABLE_CHAT);
          process.exit(0);
        }
        console.error('남은 카카오톡 창 없음, 완료.');
        break;
      }
      console.error(e.message || e);
      process.exit(1);
    }

    console.error(`[${round}] 저장 후 창 닫음 →`, outPath);

    if (once) break;

    await delay(betweenMs);
  }
}

// ─── summarize / watch ─────────────────────────────────────────────

function ensureDirs() {
  fs.mkdirSync(WATCH_DIR, { recursive: true });
  fs.mkdirSync(path.dirname(LAST_FULL_PATH), { recursive: true });
}

function readUtf8(p) {
  return fs.readFileSync(p, 'utf8');
}

function loadLastFull() {
  try {
    return readUtf8(LAST_FULL_PATH);
  } catch {
    return '';
  }
}

function saveLastFull(text) {
  fs.writeFileSync(LAST_FULL_PATH, text, 'utf8');
}

function deltaFromPrevious(fullText, previous) {
  if (!previous) return fullText;
  if (fullText.startsWith(previous)) return fullText.slice(previous.length).trimStart();
  return fullText;
}

async function summarizeChunk(text) {
  const key = process.env.GEMINI_API_KEY?.trim();
  if (!key) throw new Error('GEMINI_API_KEY 가 비어 있습니다 (.env)');

  const genAI = new GoogleGenerativeAI(key);
  const model = genAI.getGenerativeModel({ model: MODEL });
  const prompt = `다음은 카카오톡 채팅 로그의 일부(또는 전체)입니다. 한국어로 간결하게 정리하세요.

형식:
- 한 줄 제목(채팅 주제 추정)
- 핵심 요약 (불릿 3~7개)
- 결정·할 일·약속이 있으면 별도 소제목

원문에 이름이 있으면 그대로 쓰되, 불필요한 욕설·개인정보(전화·주소·계좌)는 [생략] 처리.

---
${text.slice(0, 120_000)}
---
`;
  const res = await model.generateContent(prompt);
  return res.response.text();
}

async function postDiscordWebhook(content) {
  if (!WEBHOOK) {
    console.log('\n--- 요약 (웹훅 없음) ---\n', content, '\n');
    return;
  }
  const parts = [];
  const max = 1900;
  for (let i = 0; i < content.length; i += max) parts.push(content.slice(i, i + max));
  for (const chunk of parts) {
    const r = await fetch(WEBHOOK, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content: chunk }),
    });
    if (!r.ok) {
      const t = await r.text();
      throw new Error(`Webhook ${r.status}: ${t}`);
    }
  }
}

const pending = new Map();
function debounceFile(fp, fn, ms = 600) {
  clearTimeout(pending.get(fp));
  pending.set(
    fp,
    setTimeout(() => {
      pending.delete(fp);
      fn();
    }, ms),
  );
}

async function handleTxtFile(fp) {
  let text;
  try {
    text = readUtf8(fp);
  } catch {
    return;
  }
  if (!text.trim()) return;

  const prev = loadLastFull();
  const delta = deltaFromPrevious(text, prev);
  if (!delta.trim()) {
    console.log(`[skip] 변화 없음 (접두만 동일): ${fp}`);
    return;
  }

  console.log(`[process] ${fp} (delta ${delta.length} chars, full ${text.length})`);
  let summary;
  try {
    summary = await summarizeChunk(delta);
  } catch (e) {
    console.error('[gemini]', e);
    return;
  }

  try {
    await postDiscordWebhook(`📋 **카카오 채팅 요약** (${path.basename(fp)})\n${summary}`);
  } catch (e) {
    console.error('[discord]', e);
    return;
  }

  saveLastFull(text);
  console.log('[done] 스냅샷 갱신:', LAST_FULL_PATH);
}

async function scanOnce(sinceMs) {
  ensureDirs();
  console.log('[scan] dir:', WATCH_DIR);
  console.log('[scan] since mtime:', sinceMs || '(전체)');
  console.log('State snapshot:', LAST_FULL_PATH);
  console.log('Webhook:', WEBHOOK ? 'on' : 'off');

  let names;
  try {
    names = fs.readdirSync(WATCH_DIR);
  } catch (e) {
    console.error('[scan]', e.message || e);
    process.exit(1);
    return;
  }

  const entries = [];
  for (const name of names) {
    if (!name.toLowerCase().endsWith('.txt')) continue;
    const fp = path.join(WATCH_DIR, name);
    let st;
    try {
      st = fs.statSync(fp);
    } catch {
      continue;
    }
    if (st.mtimeMs < sinceMs) continue;
    entries.push({ fp, mtime: st.mtimeMs });
  }
  entries.sort((a, b) => a.mtime - b.mtime);

  console.log(`[scan] ${entries.length} file(s) to process`);
  for (const { fp } of entries) {
    await handleTxtFile(fp);
  }
  console.log('[scan] done');
}

function startWatch() {
  ensureDirs();
  console.log('Watching:', WATCH_DIR);
  console.log('State snapshot:', LAST_FULL_PATH);
  console.log('Webhook:', WEBHOOK ? 'on' : 'off');

  fs.watch(WATCH_DIR, { persistent: true }, (_evt, name) => {
    if (!name || !name.toLowerCase().endsWith('.txt')) return;
    const fp = path.join(WATCH_DIR, name);
    debounceFile(fp, () => {
      handleTxtFile(fp).catch((e) => console.error(e));
    });
  });
}

// ─── main ───────────────────────────────────────────────────────────

async function main() {
  const { mode, scanSinceMs, triggerArgs } = partitionArgs(process.argv);
  const triggerOpts = parseTriggerArgs(triggerArgs);

  if (mode === 'watch') {
    startWatch();
    return;
  }
  if (mode === 'scan') {
    await scanOnce(scanSinceMs);
    return;
  }
  if (mode === 'trigger-only') {
    await runTrigger(triggerOpts);
    return;
  }

  const sinceMs = Date.now() - PIPELINE_SINCE_SLACK_MS;
  await runTrigger(triggerOpts);
  await scanOnce(sinceMs);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
