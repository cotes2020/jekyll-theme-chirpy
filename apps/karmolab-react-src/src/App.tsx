import { Planner } from './components/Planner';
import { HeroPanel } from './components/HeroPanel';
import { GoogleOAuthProvider, useGoogleLogin } from '@react-oauth/google';
import { useState } from 'react';
import './App.css';

const CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID || '';

function AppContent() {
  const [accessToken, setAccessToken] = useState<string | null>(null);
  const [showHero, setShowHero] = useState(false);

  const login = useGoogleLogin({
    onSuccess: (tokenResponse) => {
      console.log('Login Success!', tokenResponse);
      setAccessToken(tokenResponse.access_token);
    },
    scope: 'https://www.googleapis.com/auth/calendar.events https://www.googleapis.com/auth/tasks https://www.googleapis.com/auth/calendar.readonly',
  });

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
            <span className="app-topbar-status">✅ 연동됨</span>
          )}
        </div>
      </div>

      {/* 히어로 패널 (토글) */}
      {showHero && (
        <div className="app-hero-drawer">
          <HeroPanel />
        </div>
      )}

      {/* 메인 영역 */}
      <div className="app-main">
        {accessToken ? (
          <Planner accessToken={accessToken} />
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

export default function App() {
  return (
    <GoogleOAuthProvider clientId={CLIENT_ID}>
      <AppContent />
    </GoogleOAuthProvider>
  );
}
