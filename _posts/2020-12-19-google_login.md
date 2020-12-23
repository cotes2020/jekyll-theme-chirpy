---
title: passportjs를 이용한 google OAuth 로그인 기능 구현하기
author: juyoung
date: 2020-12-16 16:08:00 +0800
categories: [react, API]
tags: [API]
---

# 0. 구글 API console에 접속해 OAuth 클라이언트 ID를 만들기
<br />

[구글 API console](https://console.developers.google.com/apis/)에 접속해서 
사용자 인증 정보 만들기 >
OAuth 클라이언트 ID를 만든다.


![OAuth 클라이언트 ID](/assets/img/google_clientid.jpg)
<br />
<br />

# 1. frontend 서버에서 요청을 보낼 버튼 만들기
<br />
백엔드 서버 /user/google 주소로 GET요청을 보내도록 버튼을 만듭니다.
예를 들면 http://localhost:3051/user/google


components>GoogleLoginBtn.js
```javascript
import React, { useCallback } from 'react';
import { GoogleLoginButton } from 'react-social-login-buttons';
import { useRouter } from 'next/router';
import { backUrl } from '../config/config';
import { Tooltip } from 'antd';
const GoogleLoginBtn = () => {

    const router = useRouter();
    const onClickGoogleLogin = useCallback(() => {
        router.push(`${backUrl}/user/google`);

    }, []);
    return (
        <Tooltip
            placement="bottom"
            title=" 안드로이드 , iOS 및 OS X 사용자는 크롬 브라우저로 구글로 로그인하기 기능을 이용해주세요.">
            <GoogleLoginButton
                onClick={onClickGoogleLogin}
                align="center"
                size="40px"
                text="Google"
                style={{ width: '150px' }}
            />
        </Tooltip>
    );
};

export default GoogleLoginBtn;
```
<br />
<br />

# 2. backend server에서 구글 로그인 전략 짜기
<br />

.dotenv 파일에 [구글 API console](https://console.developers.google.com/apis/)에서 OAuth 2.0 클라이언트 ID로 들어가면 GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET 값을 알 수 있습니다. git에는 올리지 않도록 주의하기!<br />
callbackURL에는 받은 client 로그인 정보를 보낼 redirect url를 적어줍니다. 그러면
결과를 GET '/user/google/callback'으로 받습니다.
<br />

passport>google.js
```javascript
const passport = require('passport');
const dotenv = require('dotenv');
const bcrypt = require('bcrypt');

//구글 로그인 전략
dotenv.config();
const { User } = require('../models');
const GoogleStrategy = require('passport-google-oauth').OAuth2Strategy;

module.exports = () => {
    passport.use(new GoogleStrategy({
        clientID: process.env.GOOGLE_CLIENT_ID,
        clientSecret: process.env.GOOGLE_CLIENT_SECRET,
        callbackURL: '/user/google/callback',
    },
        async (accessToken, refreshToken, profile, done) => {
            // console.log('profile', profile);
            try {
                const exUser = await User.findOne({
                    where: {
                        email: profile.emails[0].value,
                        provider: 'google',
                    },

                });
                if (exUser) {
                    return done(null, exUser);
                }
                else {
                    const hashedPassword = await bcrypt.hash(profile.displayName, 11);
                    const newUser = await User.create({
                        email: profile.emails[0].value,
                        password: hashedPassword,
                        nickname: profile.displayName,
                        snsId: profile.id,
                        provider: 'google',
                    });
                    done(null, newUser);
                }
            }
            catch (err) {
                console.error(err);
                done(err);
            }

        }
    ));
};
```
<br />
<br />

passport>index.js
```javascript
const passport = require('passport');
const { User } = require('../models');
const local = require('./local');
const google = require('./google');
const facebook = require('./facebook');

//로그인 설정
module.exports = () => {
    passport.serializeUser((user, done) => {   
        done(null, user.id);
    });
    passport.deserializeUser(async (id, done) => {
        try {
            const user = await User.findOne({
                where: { id }
            });
            done(null, user);//req.user
        } catch (err) {
            console.error(err);
            done(err);
        }
    });
    local();
    google();
    facebook();
};

```
<br />
<br />

# 3. backend server에서 google oauth를 통해 받은 요청을 front로 다시 보낼 router 만들기
<br />

scope에는 클라이언트 로그인 정보 중 필요한 정보가 무엇인지 넣어주면 된다. 저는 'profile', 'email' 값을 요청했습니다.
GET /user/google 로 요청을 보내면 GET /google/callback으로 값을 받는다. 이를 다시 프론트 서버 주소(여기서는 frontUrl)로 보내주도록 합니다. 
<br />

passport>routes>user.js 
```javascript
const express = require('express');
const router = express.Router();
const passport = require('passport');
const prod = process.env.NODE_ENV === 'production';
const frontUrl = prod ? "https://ymillonga.xyz" : "http://localhost:3050";

router.get('/google', function (req, res, next) {// GET /user/google
    passport.authenticate('google', { scope: ['profile', 'email'] })(req, res, next);
});

router.get('/google/callback', passport.authenticate('google', {
    failureRedirect: '/',
}), async (req, res, next) => {
    return res.status(200).redirect(frontUrl);

});
module.exports = router;

```
<br />
<br />

# 4. 구글 redirection url 수정하기
<br />

![구글 403 승인오류](/assets/img/google_error.jpg)

<br />
403 에러 메세지가 나온다는 것은 승인된 redirection URL에 해당 주소값을 넣어주지 않았기 때문에 발생합니다.
<br />

아래와 같이 승인된 자바스크립트 원본(
브라우저 요청에 사용)에 프론트서버 주소를,
승인된 리디렉션 URI(웹 서버의 요청에 사용)에는 백서버 주소를 넣어줍니다.

![승인된 redirection URL](/assets/img/google_redirection_url.jpg)

참고로 2016년 10월 20일 이후로 안드로이드 , iOS 및 OS X 사용자는 크롬 브라우저로만 구글 oauth를 이용한 로그인이 가능하다.

![안드로이드 구글 403 승인오류](/assets/img/google_android_error.jpg)

참고:    

[구글 API console](https://console.developers.google.com/apis/)  

[passportjs 공식 홈페이지](http://www.passportjs.org/docs/google/)    

[인프런 react nodebird 강의](https://www.inflearn.com/course/%EB%85%B8%EB%93%9C%EB%B2%84%EB%93%9C-%EB%A6%AC%EC%95%A1%ED%8A%B8-%EB%A6%AC%EB%89%B4%EC%96%BC)  

[zerocho node-js](https://github.com/ZeroCho/nodejs-book/tree/master/ch9/9.5/nodebird)  

[403. Error: disallowed_useragent](https://spiralmoon.tistory.com/entry/Android-403-Error-disalloweduseragent)