(function () {
    function buildPlanner(container) {
        // React 앱이 마운트될 루트 엘리먼트 생성
        // container와 root 모두 full-height로 설정해야 height: 100% 체인이 동작함
        Object.assign(container.style, { height: '100%', display: 'flex', flexDirection: 'column', overflow: 'hidden', padding: '0' });
        container.innerHTML = '<div id="karmolab-planner-root" style="flex:1;min-height:0;display:flex;flex-direction:column;overflow:hidden;"></div>';

        // React 앱의 CSS 로드 (중복 방지)
        const cssUrl = '/apps/karmolab/react-dist/assets/planner.css';
        if (!document.head.querySelector(`link[href="${cssUrl}"]`)) {
            const link = document.createElement('link');
            link.rel = 'stylesheet';
            link.href = cssUrl;
            document.head.appendChild(link);
        }

        // React 앱의 JS 로드 (중복 방지)
        const jsUrl = '/apps/karmolab/react-dist/assets/planner.js';
        if (!document.body.querySelector(`script[src="${jsUrl}"]`)) {
            const script = document.createElement('script');
            script.type = 'module';
            script.src = jsUrl;
            script.onload = () => {
                if (window.mountKarmoPlanner) window.mountKarmoPlanner('karmolab-planner-root');
            };
            document.body.appendChild(script);
        } else {
            // 이미 로드된 스크립트라면 바로 렌더링 함수 재실행
            if (window.mountKarmoPlanner) window.mountKarmoPlanner('karmolab-planner-root');
        }
    }

    Toolbox.register({
        id: 'planner',
        title: '플래너',
        category: 'tool',
        desc: '나만의 일정 동기화 및 스트릭 칸반 보드',
        icon: '<rect x="3" y="4" width="18" height="18" rx="2" ry="2" fill="none" stroke="currentColor" stroke-width="2"/><line x1="16" y1="2" x2="16" y2="6" stroke="currentColor" stroke-width="2"/><line x1="8" y1="2" x2="8" y2="6" stroke="currentColor" stroke-width="2"/><line x1="3" y1="10" x2="21" y2="10" stroke="currentColor" stroke-width="2"/>',
        layout: 'full',
        noHero: true,
        tabs: [
            { id: 'planner-main', label: '대시보드', build: buildPlanner }
        ]
    });
})();
