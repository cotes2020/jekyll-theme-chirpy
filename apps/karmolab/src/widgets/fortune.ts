(function (): void {
  Toolbox.register({
    ...(Toolbox.getLazyWidgetPublicMeta?.('fortune') ?? {}),
    tabs: [
      {
        id: 'app',
        label: '운세',
        build: function (container: HTMLElement): void {
          container.innerHTML = `
                <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:380px; gap:20px; text-align:center;">
                    <div style="font-size:16px; font-weight:bold; color:var(--text-primary);">🔮 오늘의 행운을 점쳐보세요</div>
                    <div id="fortuneCookie" style="font-size:80px; cursor:pointer; transition:transform 0.2s; user-select:none;">🥠</div>
                    <div id="fortuneResult" style="font-size:15px; font-weight:500; color:var(--accent); min-height:40px; max-width:80%; line-height:1.6;">쿠키를 클릭해서 열어보세요!</div>
                    <button class="btn secondary" id="fortuneReset" style="display:none;">다시 뽑기</button>
                </div>
            `;
          const cookieEl = container.querySelector('#fortuneCookie') as HTMLElement | null;
          const resultEl = container.querySelector('#fortuneResult') as HTMLElement | null;
          const resetEl = container.querySelector('#fortuneReset') as HTMLButtonElement | null;
          if (!cookieEl || !resultEl || !resetEl) return;

          const cookie = cookieEl;
          const result = resultEl;
          const reset = resetEl;

          Mdd.linePreset('tool_run', { msg: '운명의 쿠키예요... 두근' });

          const fortunes = [
            '오늘은 집 밖으로 나가지 않는 것이 상책입니다.',
            '이불 밖은 위험합니다. 침대와 물아일체가 되세요.',
            '오늘의 행운의 장소: 당신의 방 구석지입니다.',
            '스마트폰 배터리가 유난히 빨리 닳는 하루가 될 것입니다.',
            '길을 걷다 양말이 벗겨질 수 있으니 주의하세요.',
            '오늘 산 복권은 꽝일 확률이 99.9%입니다.',
            '점심 메뉴 고민이 가장 큰 고비가 될 것입니다.',
            '택배가 올 것 같지만 안 올 것입니다.'
          ];

          cookie.onclick = () => {
            if (cookie.textContent === '🍪') return;
            cookie.style.transform = 'scale(1.2) rotate(15deg)';
            Mdd.setMood('eating');
            Mdd.bounce();
            setTimeout(() => {
              cookie.style.transform = 'scale(1) rotate(0deg)';
              cookie.textContent = '🍪';
              const rand = fortunes[Math.floor(Math.random() * fortunes.length)];
              result.innerHTML = `<span style="color:var(--error); font-weight:bold;">[대흉]</span><br>${rand}`;
              reset.style.display = 'inline-block';
              Mdd.linePreset('error', { msg: '역시... 꽝이에요... 😿' });
            }, 300);
          };

          reset.onclick = () => {
            cookie.textContent = '🥠';
            result.textContent = '쿠키를 클릭해서 열어보세요!';
            reset.style.display = 'none';
            Mdd.linePreset('tool_run', { msg: '이번엔 다를 거예요...(아닐걸)' });
          };
        }
      }
    ]
  });
})();
