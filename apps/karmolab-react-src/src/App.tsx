import { GoogleOAuthProvider, useGoogleLogin } from '@react-oauth/google';
import { useState, useEffect, lazy, Suspense } from 'react';
import './App.css';

const Planner = lazy(() =>
  import('./components/Planner').then((m) => ({ default: m.Planner }))
);
const HeroPanel = lazy(() =>
  import('./components/HeroPanel').then((m) => ({ default: m.HeroPanel }))
);

const CLIENT_ID = (import.meta.env.VITE_GOOGLE_CLIENT_ID ?? '').trim();
const STORAGE_KEY = 'karmolab_google_token';

interface TokenData {
  access_token: string;
  expires_at: number;
}

function PlannerLoadFallback() {
  return (
    <div className="app-login-prompt" aria-busy="true">
      <p className="app-login-desc">플래너를 불러오는 중…</p>
    </div>
  );
}

function AppContent() {
  const [accessToken, setAccessToken] = useState<string | null>(null);
  const [showHero, setShowHero] = useState(false);

  // 초기 로드 시 localStorage 확인
  useEffect(() => {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      try {
        const data: TokenData = JSON.parse(stored);
        const now = Date.now();
        
        // 만료되지 않은 경우에만 세션 복구 (마진 5분)
        if (data.expires_at > now + 300000) {
          setAccessToken(data.access_token);
        } else {
          localStorage.removeItem(STORAGE_KEY);
        }
      } catch (e) {
        localStorage.removeItem(STORAGE_KEY);
      }
    }
  }, []);

  const login = useGoogleLogin({
    onSuccess: (tokenResponse) => {
      console.log('Login Success!', tokenResponse);
      const expires_at = Date.now() + (tokenResponse.expires_in * 1000);
      
      const tokenData: TokenData = {
        access_token: tokenResponse.access_token,
        expires_at
      };

      setAccessToken(tokenResponse.access_token);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(tokenData));
    },
    scope: 'https://www.googleapis.com/auth/calendar.events https://www.googleapis.com/auth/tasks https://www.googleapis.com/auth/calendar.readonly',
  });

  const logout = () => {
    setAccessToken(null);
    localStorage.removeItem(STORAGE_KEY);
  };

  return (
    <div className="app-dashboard">
      {/* 상단 컴팩트 바 */}
      <div className="app-topbar">
        <div className="app-topbar-left">
          <span className="app-topbar-brand">🗓 KarmoLab Planner</span>
        </div>
        <div className="app-topbar-right">
          <button
            className="btn btn-ghost app-topbar-btn"
            onClick={() => setShowHero(v => !v)}
            title="스탯 패널"
          >
            ⚡
          </button>
          {!accessToken ? (
            <button onClick={() => login()} className="btn btn-accent app-topbar-btn">
              <img src="https://www.svgrepo.com/show/475656/google-color.svg" alt="G" className="google-icon" />
              구글 연동
            </button>
          ) : (
            <div className="app-topbar-session">
              <span className="app-topbar-status">✅ 연동됨</span>
              <button onClick={logout} className="btn btn-ghost btn-xs app-topbar-logout" title="연동 해제">
                로그아웃
              </button>
            </div>
          )}
        </div>
      </div>

      {/* 히어로 패널 (토글) */}
      {showHero && (
        <div className="app-hero-drawer">
          <Suspense fallback={<p className="app-login-desc">패널 로드 중…</p>}>
            <HeroPanel />
          </Suspense>
        </div>
      )}

      {/* 메인 영역 */}
      <div className="app-main">
        {accessToken ? (
          <Suspense fallback={<PlannerLoadFallback />}>
            <Planner accessToken={accessToken} />
          </Suspense>
        ) : (
          <div className="app-login-prompt">
            <div className="app-login-icon">🗓</div>
            <h2 className="app-login-title">KarmoLab Planner</h2>
            <p className="app-login-desc">구글 연동 후 캘린더와 할 일을 관리하세요</p>
            <button onClick={() => login()} className="btn btn-accent">
              <img src="https://www.svgrepo.com/show/475656/google-color.svg" alt="G" className="google-icon" />
              구글 계정으로 시작하기
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

/** OAuth Provider는 유효한 client_id가 있을 때만 마운트 (없으면 GIS가 즉시 예외 throw) */
function AppMissingGoogleClientId() {
  return (
    <div className="app-dashboard">
      <div className="app-topbar">
        <div className="app-topbar-left">
          <span className="app-topbar-brand">🗓 KarmoLab Planner</span>
        </div>
      </div>
      <div className="app-main">
        <div className="app-login-prompt">
          <div className="app-login-icon">⚙️</div>
          <h2 className="app-login-title">Google 클라이언트 ID 필요</h2>
          <p className="app-login-desc">
            캘린더·할 일 연동을 쓰려면 빌드 시 환경 변수 <code>VITE_GOOGLE_CLIENT_ID</code>를 설정하세요.
          </p>
          <p className="app-login-desc" style={{ fontSize: '0.9rem', opacity: 0.85 }}>
            로컬: <code>apps/karmolab-react-src/.env</code>에 OAuth 클라이언트 ID를 넣은 뒤{' '}
            <code>npm run build</code>를 다시 실행합니다. 템플릿은 <code>.env.template</code>를 참고하세요.
          </p>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  if (!CLIENT_ID) {
    return <AppMissingGoogleClientId />;
  }
  return (
    <GoogleOAuthProvider clientId={CLIENT_ID}>
      <AppContent />
    </GoogleOAuthProvider>
  );
}

