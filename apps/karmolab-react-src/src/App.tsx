import { Planner } from './components/Planner';
import { HeroPanel } from './components/HeroPanel';
import { GoogleOAuthProvider, useGoogleLogin } from '@react-oauth/google';
import { useState, useEffect } from 'react';
import './App.css';

const CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID || '';
const STORAGE_KEY = 'karmolab_google_token';

interface TokenData {
  access_token: string;
  expires_at: number;
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

